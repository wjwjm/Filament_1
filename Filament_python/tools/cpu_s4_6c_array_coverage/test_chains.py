"""Synthetic engineering only; no performance or real-input qualification."""
import gc
import weakref
import numpy as np
import pytest
import scipy.fft

from . import chains as c


def exact(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.strides == right.strides
    assert left.flags.owndata == right.flags.owndata
    assert left.flags.writeable == right.flags.writeable
    assert np.array_equal(left.view(np.uint8), right.view(np.uint8))


@pytest.mark.parametrize('workers', c.WORKERS)
@pytest.mark.parametrize('mode', ('P', 'P_LOCAL'))
def test_s46c_intensity_exact_lifetime(workers, mode):
    ref, _, _ = c.source_references(scipy.fft)
    rng = np.random.default_rng(406)
    E = np.ascontiguousarray(rng.normal(size=(64,7,9)) + 1j*rng.normal(size=(64,7,9)))
    before = E.copy(); old = ref(E, 1.0003)
    with c.Workers(workers) as pool:
        got = c.intensity(E, 1.0003, pool, mode)
        exact(old, got); exact(E, before)
        assert not np.shares_memory(E, got)
        keep = got.copy()
        c.intensity(E * 2, 1.0003, pool, mode)
        exact(got, keep)
        rows = pool.take_records()
        assert all(r['actual_workers'] == workers for r in rows)
        weak = weakref.ref(E)
        del E
        gc.collect()
        assert weak() is None
    with pytest.raises(ValueError, match='RETIRED'):
        pool.run('invalid', 1, lambda *_: None)


@pytest.mark.parametrize('workers', c.WORKERS)
@pytest.mark.parametrize('mode', ('P', 'P_LOCAL'))
def test_s46c_eq10_exact_global_boundaries(workers, mode):
    _, ref, _ = c.source_references(scipy.fft)
    rng=np.random.default_rng(407)
    I=np.ascontiguousarray(np.abs(rng.normal(size=(64,7,9))))
    response=np.ascontiguousarray(rng.normal(size=I.shape))
    omega=np.asarray(np.fft.fftfreq(64),dtype=np.float64)
    before=(I.copy(),response.copy(),omega.copy())
    expected=ref(I,response,omega,0.13,2.1e-23)
    class FFT:
        def __init__(self): self.calls=[]
        def fft(self,a,axis): self.calls.append(('fft',a.shape,axis));return scipy.fft.fft(a,axis=axis)
        def ifft(self,a,axis): self.calls.append(('ifft',a.shape,axis));return scipy.fft.ifft(a,axis=axis)
    fft=FFT()
    with c.Workers(workers) as pool:
        got=c.eq10(I,response,omega,0.13,2.1e-23,fft,pool,mode)
        for k in expected: exact(expected[k],got[k])
        assert fft.calls==[('fft',I.shape,0),('ifft',I.shape,0)]
        assert all(r['actual_workers']==workers for r in pool.take_records())
    for left,right in zip((I,response,omega),before): exact(left,right)


@pytest.mark.parametrize('mode', ('P','P_LOCAL'))
def test_s46c_extreme_signzero_nan_underflow_and_refusals(mode):
    ref_a,ref_b,_=c.source_references(scipy.fft)
    vals=np.array([0.,-0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),1e-160,-1e-160,1e154,-1e154,np.inf,-np.inf,np.nan])
    E=np.empty((64,2,len(vals)),dtype=np.complex128); E.real=vals;E.imag=-vals
    with c.Workers(1) as pool, np.errstate(all='ignore'):
        exact(ref_a(E,1.),c.intensity(E,1.,pool,mode))
        for value in (0.,-0.,np.nextafter(0.,1.),np.nan):
            I=np.full((64,2,3),value,dtype=np.float64);R=np.ones_like(I);O=np.fft.fftfreq(64)
            ref=ref_b(I,R,O,1.,1e-23);got=c.eq10(I,R,O,1.,1e-23,scipy.fft,pool,mode)
            for k in ref: exact(ref[k],got[k])
        with pytest.raises(ValueError,match='ARRAY_DOMAIN'):
            c.intensity(E.astype(np.complex64),1.,pool,mode)
        with pytest.raises(ValueError,match='UNSUPPORTED_NATIVE_LAYOUT'):
            c.intensity(np.asfortranarray(E),1.,pool,mode)
        with pytest.raises(ValueError,match='OMEGA_DOMAIN'):
            c.eq10(I,R,O[:-1],1.,1e-23,scipy.fft,pool,mode)


def test_s46c_configuration_and_overlapping_output_refused():
    for workers in (0,2,True,96):
        with pytest.raises(ValueError,match='WORKER_CONFIGURATION'): c.Workers(workers)
    array=np.ones((64,2,3))
    with c.Workers(1) as pool:
        with pytest.raises(ValueError,match='INPUT_OUTPUT_ALIAS'):
            pool.ufunc('bad',np.multiply,(array,2.),array)
