"""GPU manager simulation only: no CUDA or GPU mathematical qualification."""
from types import SimpleNamespace
from .gpu import Meter


class Cuda:
    def __init__(self):
        self.syncs=0; self.tick=0; self.runtime=self
    def deviceSynchronize(self):self.syncs+=1
    def get_current_stream(self):return SimpleNamespace(ptr=7)
    def Event(self):
        parent=self
        class Event:
            def record(self):
                parent.tick+=1; self.tick=parent.tick
        return Event()
    def get_elapsed_time(self,start,end):return float(end.tick-start.tick)


def test_s46a_gpu_completion_anchors_and_deferred_event_read():
    for mode in ('OFF','LIGHT'):
        cuda=Cuda(); meter=Meter(SimpleNamespace(cuda=cuda),mode,8)
        for index in range(8):
            with meter.scope('INTERVAL_BODY',index):
                with meter.scope('LINEAR'):
                    pass
        assert cuda.syncs==2
        result=meter.finish()
        assert cuda.syncs==3
        bodies=[row for row in result['records'] if row['label']=='INTERVAL_BODY']
        assert len(bodies)==8 and [row['interval'] for row in bodies]==list(range(8))
        assert all(row['completion_anchor']==(row['interval'] in (0,7)) for row in bodies)
        assert ('device_event_span_ms' in bodies[0])==(mode=='LIGHT')
        assert result['kernel_activity']=='UNKNOWN_WITHOUT_TRACE'


def test_s46a_gpu_original_ast_compiles_without_gpu_math(tmp_path):
    from .gpu import build
    for mode in ('REFERENCE','OFF','LIGHT'):
        meter=Meter(SimpleNamespace(cuda=Cuda()),mode,8)
        function,binding,captures,transfers,calls=build(meter,tmp_path/mode,False)
        assert callable(function) and binding['timer_stripping_restores_source']
        assert binding['scientific_expression_changes']==0 and not captures and not transfers
