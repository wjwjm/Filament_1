"""TEST_FIXTURE_ONLY: deterministic optical double, real Batch/Streaming hydro.

Invoked by the repository test wrapper's subprocess test. Never a HPC launcher.
The real optical adapter is qualified separately by the tiny propagation test.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np

from KHz_filament.hr4c_state import HR4CThreeFieldStore, evolve_hr4_full_z
from KHz_filament.hr4d_pulse_lifecycle import build_interpulse_step_schedule
from KHz_filament.hr4e5_evidence import atomic_json, compare_arrays_exact, sha256_array, sha256_file
from KHz_filament.hr4e5_formal_entry import build_fixture_admission_identity, create_successor_root, open_successor_root, validate_final_post
from KHz_filament.hr4e5_paired_campaign import FormalPairedDriver, PairedCampaign
from KHz_filament.hr4e5_storage import MockQuotaProvider
from KHz_filament.hr4e5s_streaming import StreamingLifecycle, FIELDS

K, SHAPE, N = 32, (8, 8), 3
SINKS = ('ion', 'ib', 'raman', 'qthermal', 'increment', 'state_after')
LEDGERS = ('E_dep_ion_interval_J', 'E_dep_ib_interval_J', 'E_dep_raman_interval_J',
           'E_dep_plasma_interval_J', 'E_thermal_interval_J', 'delta_n_increment_min',
           'delta_n_increment_onaxis', 'delta_n_state_min_after_update', 'delta_n_state_onaxis_after_update')
HYDRO = dict(dt_hydro=1e-7, n_hydro_steps=2, chi=1e-5, nu=1e-5, n0=1.00027,
             gravity_x=0., gravity_y=0., cfl_limit=1.)

def records():
    return [dict(ordinal=i, screen_id=f'fixture-{i}', z_m=(i+.5)*1e-4) for i in range(K)]

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def initial():
    y, x = np.indices(SHAPE, dtype=np.float64)
    profile = np.sin(np.pi*x/7)*np.sin(np.pi*y/7)
    return dict(delta_n=np.stack([-1e-7*(i+1)*profile for i in range(K)]),
                vx=np.stack([1e-4*(i+1)*profile for i in range(K)]),
                vy=np.stack([-2e-4*(i+1)*profile for i in range(K)]))

class Fixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        new = not (self.root/'E5_1A_CAMPAIGN_STATE.json').exists()
        self.admission_identity = build_fixture_admission_identity(campaign_id='TEST_FIXTURE_ONLY', n_pulses=N, k=K, shape=SHAPE, epoch='fixture')
        self.epoch = f'fixture:{os.getpid()}:{uuid.uuid4().hex}'
        self.campaign = PairedCampaign(self.root, create=new, n_pulses=N,
            campaign_id='TEST_FIXTURE_ONLY', safety_margin_bytes=1024*1024,
            admission_identity=self.admission_identity)
        self.budget = self.campaign.storage
        self.budget.provider = MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40)
        if new:
            with self.writes('initialize'):
                np.save(self.root/'source.npy', np.arange(512, dtype=np.float64).reshape(8,8,8).astype(np.complex128))
                r = self.track('R', 0); r.mkdir(parents=True)
                for f, v in initial().items(): np.save(r/f'pre_{f}.npy', v)
                StreamingLifecycle.create(root=self.track('C',0)/'state', current=initial(),
                    screen_records=records(), current_generation='fixture:C:p0', dx_m=1e-4, dy_m=1e-4)
                atomic_json(r/'READY.json', self.reference_ready(r))

    def track(self, side, p): return self.root/side/f'p{p}'

    def writes(self, label):
        # Bounded fixture uses phase-level admission before invoking existing writers.
        # A failed phase retains its arrays and reservation; it never auto-cleans.
        from contextlib import contextmanager
        @contextmanager
        def context():
            before = {p for p in self.root.rglob('*') if p.is_file()}
            token = self.budget.reserve(16*1024**2, purpose=label, owner=self.epoch)
            yield
            for path in self.root.rglob('*'):
                if path.is_file() and path not in before and path.suffix in ('.npy','.npz'):
                    final = path.name in ('final_optical_field.npy','scientific_ledger.npz','source.npy')
                    self.budget.register_artifact(path, role='final' if final else 'intermediate',
                        trajectory=path.relative_to(self.root).parts[0], attempt=0,
                        reclaimable=not final, expected_sha256=sha256_file(path),
                        metadata={'creation_phase':label, 'writer_epoch':self.epoch})
            self.budget.consume(token.reservation_id)
        return context()

    def reference_ready(self, root):
        fields = {}
        for f in FIELDS:
            path = root/f'pre_{f}.npy'; a=np.load(path, allow_pickle=False)
            if a.shape != (K,*SHAPE) or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError('invalid independent reference PRE')
            fields[f] = dict(path=path.name, sha256=sha256_file(path), array_hash=sha256_array(a))
        ready = dict(status='READY', schema='fixture.reference.ready.v1', fields=fields,
                     child_root=str(root.resolve()), self_contained=True,
                     parent_payload_required=False, scope='TEST_FIXTURE_ONLY')
        binding = root/'binding.json'
        if binding.is_file():
            ready['binding'] = dict(path=str(binding.resolve()), sha256=sha256_file(binding))
        return ready

    def optical(self, root, pre_reader, post_writer):
        source=np.load(self.root/'source.npy',allow_pickle=False); fresh=source.copy()
        assert not np.shares_memory(source,fresh)
        sinks={name:np.empty((K,*SHAPE),dtype=np.float64) for name in SINKS}
        ledgers={name:np.empty(K,dtype=np.float64) for name in LEDGERS}
        pre_hashes=[]
        for i in range(K):
            pre=pre_reader(i); pre_hashes.append({f:sha256_array(pre[f]) for f in FIELDS})
            ion=np.full(SHAPE,(i+1)*1e-6,dtype=np.float64)
            ib=ion*0.5; raman=ion*0.25; q=ion+ib+raman
            increment=-q*1e-4; after=pre['delta_n']+increment
            post={'delta_n':after,'vx':pre['vx'].copy(),'vy':pre['vy'].copy()}
            post_writer(i,post)
            for name,a in zip(SINKS,(ion,ib,raman,q,increment,after)): sinks[name][i]=a
            values=(ion.sum(),ib.sum(),raman.sum(),(ion+ib).sum(),q.sum(),increment.min(),increment[4,4],after.min(),after[4,4])
            for name,value in zip(LEDGERS,values): ledgers[name][i]=value
            fresh += complex(float(after.sum()),0)
        for name,a in sinks.items(): np.save(root/f'sink_{name}.npy',a)
        np.save(root/'final_optical_field.npy',fresh)
        np.savez(root/'scientific_ledger.npz',**ledgers)
        atomic_json(root/'optical_run.json',dict(status='PASS', optical='DETERMINISTIC_TEST_DOUBLE',
            source_hash=sha256_array(source), source_alias=False, pre_hashes=pre_hashes,
            final_hash=sha256_array(fresh), ledger_fields=list(ledgers)))

    def reference(self,p):
        root=self.track('R',p)
        if read_json(root/'READY.json') != self.reference_ready(root): raise ValueError('reference READY changed')
        pre={f:np.load(root/f'pre_{f}.npy',allow_pickle=False) for f in FIELDS}
        post={f:np.empty_like(a) for f,a in pre.items()}
        with self.writes(f'R{p}'):
            self.optical(root,lambda i:{f:a[i] for f,a in pre.items()},
                lambda i,fields:[post[f].__setitem__(i,fields[f]) for f in FIELDS])
            for f,a in post.items(): np.save(root/f'post_{f}.npy',a)
            if p<N-1:
                store=HR4CThreeFieldStore(output_path=str(root/'batch'),n_intervals=K,shape=SHAPE,
                    dtype=np.float64,z_edges=np.arange(K+1)*1e-4,dx=1e-4,dy=1e-4,
                    authoritative_metadata={'scope':'TEST_FIXTURE_ONLY_REFERENCE'})
                try:
                    store.begin_staging();store.write_staging_batch(0,post);store.commit_staging({'operation':'fixture_post','batch_intervals':8})
                    evolve_hr4_full_z(store,batch_intervals=8,**HYDRO)
                    out=store.read_authoritative_batch(0,K)
                    for f,a in out.items(): np.save(root/f'next_{f}.npy',a)
                finally: store.close()
        return dict(status='PASS',root=str(root))

    def candidate(self,p):
        root=self.track('C',p); state=root/'state'
        lifecycle=StreamingLifecycle.open(state) if p==0 else open_successor_root(state)
        errors=[]; available=threading.Event(); finished=threading.Event()
        def consumer():
            try:
                worker=StreamingLifecycle.open(state)
                while True:
                    available.clear()
                    block=worker.run_one_hydro_block(**HYDRO,actor='fixture_hydro')
                    if block: continue
                    if finished.is_set(): break
                    if not available.wait(30): raise TimeoutError('fixture producer stopped')
            except BaseException as e: errors.append(e)
        def post(i,fields):
            lifecycle.deposition_finalized(i);lifecycle.commit_post(i,fields)
            if p<N-1:
                available.set()
                lifecycle.enqueue_post(i,wait_for_capacity=True,timeout_s=30)
                available.set()
        with self.writes(f'C{p}'):
            lifecycle.record_telemetry('OPTICAL_START',actor='fixture_optical')
            thread=threading.Thread(target=consumer) if p<N-1 else None
            if thread: thread.start()
            try:
                self.optical(root,lifecycle.current_fields,post)
                lifecycle.record_telemetry('OPTICAL_COMPLETE',actor='fixture_optical')
            finally:
                finished.set();available.set()
                if thread: thread.join(60)
            if thread and thread.is_alive(): raise TimeoutError('hydro writer did not exit')
            if errors: raise errors[0]
            lifecycle=StreamingLifecycle.open(state)
            if p<N-1:
                lifecycle.validate_barrier();lifecycle.promote_next_to_current()
            else:
                optical_run = read_json(root/'optical_run.json')
                optical_run.update(final=True, lifecycle_root=str(state.resolve()),
                    current_generation=lifecycle.manifest['current_generation'],
                    current_content_sha256=lifecycle.manifest['current_content_sha256'],
                    schedule_intervals=K)
                atomic_json(root/'optical_run.json', optical_run)
                result=validate_final_post(lifecycle_root=state,receipt_path=state/'POST_FINAL_READY.json',
                    writer_quiescent=True,fixture_only=True,expected_optical_dir=root)
                if result['status']!='PASS': raise ValueError(result)
            atomic_json(root/'writer_closed.json',dict(status='PASS',campaign_id='TEST_FIXTURE_ONLY',
                active_writers=[],writer_epoch=self.epoch,coordinator_process_id=self.epoch))
        return dict(status='PASS',root=str(root))

    def compare(self,p,*unused):
        rows=[];r=self.track('R',p);c=self.track('C',p);lc=StreamingLifecycle.open(c/'state')
        with self.writes(f'exact{p}'):
            for namespace in ('pre','post')+(() if p==N-1 else ('next',)):
                for f in FIELDS:
                    rp=r/f'{namespace}_{f}.npy'; left=np.load(rp,mmap_mode='r',allow_pickle=False)
                    for i in range(K):
                        entry=lc.manifest['records'][i]['current' if namespace=='pre' else namespace]
                        cp=c/'state'/entry['artifact']
                        with np.load(cp,allow_pickle=False) as z:
                            rows.append(compare_arrays_exact(left[i],z[f],name=f'{namespace}:{i}:{f}',reference_path=rp,candidate_path=cp))
                    del left
            for name in SINKS:
                rp=r/f'sink_{name}.npy';cp=c/f'sink_{name}.npy'
                a=np.load(rp,mmap_mode='r');b=np.load(cp,mmap_mode='r')
                for i in range(K): rows.append(compare_arrays_exact(a[i],b[i],name=f'{name}:{i}',reference_path=rp,candidate_path=cp))
                del a,b
            screen_count=len(rows); expected=(12 if p==N-1 else 15)*K
            assert screen_count==expected
            with np.load(r/'scientific_ledger.npz') as a,np.load(c/'scientific_ledger.npz') as b:
                assert set(a.files)==set(b.files)==set(LEDGERS)
                for name in LEDGERS: rows.append(compare_arrays_exact(a[name],b[name],name=name))
            rows.append(compare_arrays_exact(np.load(r/'final_optical_field.npy'),np.load(c/'final_optical_field.npy'),name='final_optical'))
            assert all(x['status']=='PASS' for x in rows)
            report=dict(status='PASS',screen_count=screen_count,ledger_count=9,optical_count=1,rows=rows,
                mismatch_count=sum(row['status']!='PASS' for row in rows),
                compared_object_count=len(rows), expected_object_count=expected+10,
                missing_reference=[], missing_candidate=[])
            atomic_json(self.root/f'exact{p}.json',report)
        return {k:v for k,v in report.items() if k!='rows'}

    def terminal(self, p):
        receipt = self.track('C', p) / 'state' / 'POST_FINAL_READY.json'
        if not receipt.is_file() or read_json(receipt).get('terminal') != 'POST_FINAL_READY':
            raise ValueError('terminal POST_FINAL_READY receipt is missing')
        return dict(status='PASS', report_path=str(receipt), scope='TEST_FIXTURE_ONLY')

    def reclaim(self,side,p,ready):
        root=self.track(side,p)
        if ready is None:
            if p == N-1:
                ready_path = self.track('C',p)/'state'/'POST_FINAL_READY.json'
            elif side == 'C':
                ready_path = self.track(side,p+1)/'state'/'E5_1A_READY.json'
            else:
                ready_path = self.track(side,p+1)/'READY.json'
            ready = read_json(ready_path)
        targets=[]
        for rel,item in self.budget.artifacts().items():
            path=self.root/rel
            if path.is_relative_to(root) and item['reclaimable']:
                if side=='C' and p==N-1 and path.parent.name in ('current','post'): continue
                targets.append(rel)
        writer=self.root/f'quiescent_{side}{p}.json'
        atomic_json(writer,dict(status='PASS',campaign_id='TEST_FIXTURE_ONLY',active_writers=[],
            writer_epoch=self.epoch,coordinator_process_id=self.epoch))
        assert read_json(self.root/f'exact{p}.json')['status']=='PASS'
        assert ready['status'] in ('PASS','READY')
        exact_path = self.root/f'exact{p}.json'
        if p == N-1:
            ready_path = self.track('C',p)/'state'/'POST_FINAL_READY.json'
        elif side == 'C':
            ready_path = self.track(side,p+1)/'state'/'E5_1A_READY.json'
        else:
            ready_path = self.track(side,p+1)/'READY.json'
        dependency_path = self.root/f'gc_dependency_{side}{p}.json'
        atomic_json(dependency_path, dict(status='PASS', no_future_dependency=True,
            campaign_id='TEST_FIXTURE_ONLY', target_root=str(root.resolve())))
        target_bindings = []
        for relative,item in self.budget.artifacts().items():
            if relative in targets:
                target_bindings.append(dict(relative_path=relative,
                    identity=dict(item['identity']), sha256=item['sha256']))
        prerequisite_path = self.root/f'gc_prerequisites_{side}{p}.json'
        atomic_json(prerequisite_path, dict(
            schema='khz_filament.hr4e5.e5_1a.reclaim_prerequisites.v1',
            status='PASS', campaign_id='TEST_FIXTURE_ONLY',
            gates=dict(exact_complete=True, successor_ready=True,
                       no_active_writers=True, no_future_dependency=True,
                       receipts_durable=True), target_bindings=target_bindings,
            evidence=dict(
                exact_complete=dict(path=str(exact_path.resolve()), sha256=sha256_file(exact_path)),
                successor_ready=dict(path=str(ready_path.resolve()), sha256=sha256_file(ready_path)),
                no_future_dependency=dict(path=str(dependency_path.resolve()), sha256=sha256_file(dependency_path)),
            )))
        plan=self.budget.plan_reclaim(targets,exact_complete=True,successor_ready=True,no_active_writers=True,
            no_future_dependency=True,receipts_durable=True,writer_receipt=writer,
            prerequisite_receipt=prerequisite_path,plan_id=f'{side}{p}')
        self.budget.apply_reclaim(plan['plan_id'])
        assert self.budget.verify_reclaim(plan['plan_id'])['status']=='PASS'
        return dict(status='PASS', plan_id=f'{side}{p}', scope='TEST_FIXTURE_ONLY')

    def handoff(self,p,side=None,*unused):
        # Crucially R child is built and R parent reclaimed BEFORE C child copy.
        sides = (str(side),) if side is not None else ('R','C')
        ready_by_side = {}
        for side in sides:
            root=self.track(side,p)
            if p==N-1:
                ready=read_json(self.track('C',p)/'state'/'POST_FINAL_READY.json')
            else:
                child=self.track(side,p+1)
                with self.writes(f'{side}{p}_handoff'):
                    if side=='R':
                        child.mkdir(parents=True);rows=[]
                        for f in FIELDS:
                            a=np.load(root/f'next_{f}.npy');np.save(child/f'pre_{f}.npy',a)
                            b=np.load(child/f'pre_{f}.npy')
                            for i in range(K):rows.append(compare_arrays_exact(a[i],b[i],name=f'{i}:{f}'))
                        assert all(x['status']=='PASS' for x in rows)
                        atomic_json(child/'binding.json',dict(status='PASS',rows=rows))
                        ready=self.reference_ready(child);atomic_json(child/'READY.json',ready)
                    else:
                        lc,ready=create_successor_root(parent_root=root/'state',child_root=child/'state',fixture_only=True,admission_identity=self.admission_identity)
                        open_successor_root(lc.root)
            atomic_json(root/'ARCHIVED.json',dict(status='ARCHIVED_AFTER_EXACT',restartable=False,
                successor=None if p==N-1 else str(self.track(side,p+1)))) if not(side=='C' and p==N-1) else None
            ready_by_side[side] = ready
        return dict(status='PASS',sequential_handoffs=True, side=None if side is None else str(side),
                    ready=ready_by_side)

    def run(self,stop):
        schedule=build_interpulse_step_schedule(f_rep=5e6,dt_hydro=HYDRO['dt_hydro'])
        assert schedule.remainder_s==0 and schedule.full_step_count==HYDRO['n_hydro_steps']
        driver = FormalPairedDriver(self.root, admission_identity=self.admission_identity,
                                    runner=self, n_pulses=N, campaign_id='TEST_FIXTURE_ONLY',
                                    safety_margin_bytes=1024*1024, fixture_only=True)
        report = driver.run(stop_after=min(stop,N))
        atomic_json(self.root/f'process_{os.getpid()}.json',report)
        print(json.dumps(dict(pid=os.getpid(),epoch=report['epoch'],next_pair=report['next_pair_index'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--stop',type=int,default=3)
    args=parser.parse_args();Fixture(args.root).run(args.stop)
