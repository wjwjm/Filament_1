"""Separate thread coverage from local intermediate materialization.

The reference bodies are extracted from the frozen Raman source. P preserves
global intermediates and operation barriers. P_LOCAL uses the same ufuncs in
the same per-element order, with bounded scratch. Reductions and FFT remain
whole-array calls. Every returned output is newly owned; no input is written.
"""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import threading
import time

import numpy as np

VERSION = 'S46C_ARRAY_CHAINS_V1'
WORKERS = (1, 8, 24, 48)
BLOCK_ELEMENTS = 16384


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def native(value, dtype, name):
    require(type(value) is np.ndarray and value.ndim == 3 and value.dtype == np.dtype(dtype),
            'ARRAY_DOMAIN:' + name)
    require(value.flags.c_contiguous, 'UNSUPPORTED_NATIVE_LAYOUT:' + name)


def source_references(fft):
    """Explicitly bind two complete local reference chains to frozen ASTs."""
    from tools.cpu_s4_3c.candidate import _source_identity
    binding = _source_identity()
    raman = importlib.import_module('KHz_filament.raman')
    require(raman.xp is np, 'NUMPY_ONLY')
    source = Path(raman.__file__).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    stage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'isaacs_raman_stage')
    intensity = next(n for n in stage.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'intensity')
    energy = next(n for n in stage.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == 'return_energy')
    template = ast.parse('def intensity_ref(E, n0):\n rdtype=xp.float64\n pass\n return intensity\n'
                         'def eq10_ref(intensity, response, Omega, dt, n_R):\n rdtype=xp.float64\n result={}\n pass\n return result\n')
    template.body[0].body[1:2] = [copy.deepcopy(intensity)]
    template.body[1].body[2:3] = copy.deepcopy(energy.body)
    # Only FFT dispatch differs from the production module's NumPy namespace:
    # it is the actual source-bound L1 SciPy workers48 dispatcher supplied by caller.
    class XP:
        def __getattr__(self, name):
            return fft if name == 'fft' else getattr(np, name)
    namespace = dict(vars(raman), xp=XP())
    exec(compile(ast.fix_missing_locations(template), str(Path(__file__).resolve()), 'exec'), namespace)
    return namespace['intensity_ref'], namespace['eq10_ref'], binding


class Workers:
    """Persistent, bounded threads; only scalar activity records survive a call."""
    def __init__(self, count):
        require(type(count) is int and count in WORKERS, 'WORKER_CONFIGURATION')
        self.count = count
        self.closed = False
        self.records = []
        self._lock = threading.Lock()
        started = time.perf_counter_ns()
        self.executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix='s46c-array')
        barrier = threading.Barrier(count)
        def prime():
            barrier.wait(timeout=60)
            return threading.get_native_id()
        jobs = [self.executor.submit(prime) for _ in range(count)]
        self.initial_thread_ids = [j.result() for j in jobs]
        require(len(set(self.initial_thread_ids)) == count, 'POOL_NOT_FULLY_STARTED')
        self.initialization_ns = time.perf_counter_ns() - started

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def take_records(self):
        rows, self.records = self.records, []
        return rows

    def run(self, label, length, operation):
        require(not self.closed, 'RETIRED_WORKER_POOL')
        require(self._lock.acquire(blocking=False), 'CONCURRENT_OR_REENTRANT_CHAIN')
        try:
            barrier = threading.Barrier(self.count)
            errors = np.geterr()
            def worker(i):
                lo, hi = length * i // self.count, length * (i + 1) // self.count
                barrier.wait(timeout=60)
                wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
                with np.errstate(**errors):
                    operation(lo, hi)
                return dict(thread_id=threading.get_native_id(), items=hi-lo,
                            wall_ns=time.perf_counter_ns()-wall, thread_cpu_ns=time.thread_time_ns()-cpu)
            jobs = [self.executor.submit(worker, i) for i in range(self.count)]
            # Join every submitted task before releasing the scope, including
            # on a failed ufunc. No task may outlive the owning call.
            from concurrent.futures import wait
            wait(jobs)
            rows = [j.result() for j in jobs]
            self.records.append(dict(label=label, requested_workers=self.count,
                                     actual_workers=len({r['thread_id'] for r in rows if r['items']}), workers=rows))
        finally:
            self._lock.release()

    def ufunc(self, label, function, operands, output):
        require(function in (np.absolute, np.square, np.multiply, np.divide), 'UFUNC_ALLOWLIST')
        require(output.flags.c_contiguous and output.flags.owndata, 'OWNED_C_OUTPUT')
        for a in operands:
            if isinstance(a, np.ndarray):
                require(not np.shares_memory(a, output), 'INPUT_OUTPUT_ALIAS')
        # All registered real target arrays have C order; split axis zero.
        def part(a, lo, hi):
            if isinstance(a, np.ndarray) and a.ndim == output.ndim and a.shape[0] == output.shape[0]:
                return a[lo:hi]
            return a
        def operation(lo, hi):
            function(*(part(a, lo, hi) for a in operands), out=output[lo:hi])
        self.run(label, output.shape[0], operation)


def intensity(E, n0, pool, mode):
    native(E, 'complex128', 'E')
    require(mode in ('P', 'P_LOCAL'), 'MODE')
    from KHz_filament.constants import eps0, c0
    coefficient = 0.5 * float(eps0) * float(c0) * float(n0)
    if mode == 'P':
        magnitude = np.empty(E.shape, dtype=np.float64)
        pool.ufunc('A_ABS', np.absolute, (E,), magnitude)
        squared = np.empty(E.shape, dtype=np.float64)
        pool.ufunc('A_SQUARE', np.square, (magnitude,), squared)
        del magnitude
        result = np.empty(E.shape, dtype=np.float64)
        pool.ufunc('A_SCALE', np.multiply, (coefficient, squared), result)
        return result.astype(np.float64, copy=False)
    result = np.empty(E.shape, dtype=np.float64)
    src, dest = E.reshape(-1), result.reshape(-1)
    require(np.shares_memory(src, E) and np.shares_memory(dest, result), 'NO_LAYOUT_COPY')
    def operation(lo, hi):
        magnitude = np.empty(min(BLOCK_ELEMENTS, hi-lo), dtype=np.float64)
        squared = np.empty_like(magnitude)
        for start in range(lo, hi, BLOCK_ELEMENTS):
            end = min(start + BLOCK_ELEMENTS, hi)
            mag, sq = magnitude[:end-start], squared[:end-start]
            np.absolute(src[start:end], out=mag)
            np.square(mag, out=sq)
            np.multiply(coefficient, sq, out=dest[start:end])
    pool.run('A_LOCAL_ABS_SQUARE_SCALE', E.size, operation)
    return result.astype(np.float64, copy=False)


def eq10(intensity, response, Omega, dt, n_R, fft, pool, mode):
    native(intensity, 'float64', 'intensity')
    native(response, 'float64', 'response')
    require(intensity.shape == response.shape, 'INPUT_SHAPES')
    require(type(Omega) is np.ndarray and Omega.dtype == np.float64
            and Omega.shape == (intensity.shape[0],), 'OMEGA_DOMAIN')
    require(mode in ('P', 'P_LOCAL'), 'MODE')
    from KHz_filament.constants import c0
    # Original global abs/max, FFT, IFFT and sum; no blockwise reductions.
    scale = float(np.max(np.abs(intensity)))
    if scale > 0.0:
        multiplier = (1j * np.asarray(Omega, dtype=np.float64) * float(dt))[:, None, None]
        spectrum = fft.fft(intensity, axis=0)
        native(spectrum, 'complex128', 'spectrum')
        weighted = np.empty_like(spectrum)
        pool.ufunc('B_SPECTRAL_MULTIPLY', np.multiply, (multiplier, spectrum), weighted)
        del spectrum
        inverse = fft.ifft(weighted, axis=0)
        del weighted
        native(inverse, 'complex128', 'inverse')
        if mode == 'P':
            derivative = np.empty(intensity.shape, dtype=np.float64)
            pool.ufunc('B_DERIVATIVE_DIVIDE', np.divide, (inverse.real, scale), derivative)
            del inverse
            normalized_response = np.empty_like(response)
            pool.ufunc('B_RESPONSE_DIVIDE', np.divide, (response, scale), normalized_response)
            product = np.empty_like(response)
            pool.ufunc('B_PRODUCT', np.multiply, (normalized_response, derivative), product)
            del derivative, normalized_response
        else:
            product = np.empty_like(response)
            real = inverse.real.reshape(-1)
            resp, target = response.reshape(-1), product.reshape(-1)
            require(np.shares_memory(real, inverse) and np.shares_memory(resp, response), 'NO_LAYOUT_COPY')
            def operation(lo, hi):
                derivative = np.empty(min(BLOCK_ELEMENTS, hi-lo), dtype=np.float64)
                normalized_response = np.empty_like(derivative)
                for start in range(lo, hi, BLOCK_ELEMENTS):
                    end = min(start + BLOCK_ELEMENTS, hi)
                    d, r = derivative[:end-start], normalized_response[:end-start]
                    np.divide(real[start:end], scale, out=d)
                    np.divide(resp[start:end], scale, out=r)
                    np.multiply(r, d, out=target[start:end])
            pool.run('B_LOCAL_DIVIDE_DIVIDE_PRODUCT', product.size, operation)
            del inverse, real
        energy_scale = (float(n_R) / float(c0)) * scale * scale
        signed = energy_scale * np.sum(product, axis=0)
    else:
        signed = np.zeros(intensity.shape[1:], dtype=np.float64)
    return dict(u_R_signed=signed.astype(np.float64, copy=False),
                q_R_positive=np.maximum(-signed, 0.0).astype(np.float64, copy=False))
