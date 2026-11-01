"""One eager forward; unchanged Reduced reductions plus complete attention vector."""
import time
from contextlib import contextmanager
from reduced_memtrace_code_attack.features import reduce_hidden, confidence_features, combine_features
from attention_memtrace_code_attack.features import reduce_attentions


@contextmanager
def attention_timer(model, cuda):
    """Public forward hooks time attention modules, never replace/alter their outputs."""
    import torch
    events, handles = [], []
    if cuda:
        for layer in model.model.layers:
            def before(module, args):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events.append([event,None])
            def after(module, args, output):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[-1][1] = event
            handles.extend([layer.self_attn.register_forward_pre_hook(before), layer.self_attn.register_forward_hook(after)])
    try:
        yield events
    finally:
        for handle in handles:
            handle.remove()


class Extractor:
    def __init__(self, model, schema):
        self.model, self.schema = model, schema
        if model.config._attn_implementation != 'eager':
            raise ValueError('Complete attention extraction requires explicitly loaded eager backend')
        model.eval()
        model.requires_grad_(False)

    def one(self, plan):
        import numpy as np
        import torch
        device = self.model.device
        cuda = device.type == 'cuda'
        def sync():
            if cuda:
                torch.cuda.synchronize(device)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)/2**30 if cuda else 0.
        started = time.perf_counter()
        prefix = plan.get('verified_reduced_features')
        if prefix is not None and (len(prefix) != len(self.schema['reduced']['names']) or not np.isfinite(prefix).all()):
            raise ValueError('Invalid verified Reduced prefix')
        with torch.inference_mode():
            ids = torch.tensor([plan['input_ids']], device=device, dtype=torch.long)
            mask = torch.tensor([plan['attention_mask']], device=device, dtype=torch.long)
            positions = torch.tensor(plan['target_positions'], device=device, dtype=torch.long)
            forward_start = time.perf_counter()
            with attention_timer(self.model, cuda) as events:
                output = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=prefix is None,
                    output_attentions=True, use_cache=False, return_dict=True)
            sync()
            forward_seconds = time.perf_counter()-forward_start
            module_seconds = sum(a.elapsed_time(b) for a,b in events)/1000 if cuda else None
            if output.attentions is None or len(output.attentions) != self.schema['attention']['layers']:
                raise RuntimeError('Missing attention maps; refusing to drop any attention feature')
            shapes = dict(hidden_states=[list(h.shape) for h in output.hidden_states] if output.hidden_states is not None else [],
                logits=list(output.logits.shape), attentions=[list(a.shape) if a is not None else None for a in output.attentions],
                attention_tensor_count=len(output.attentions))
            reduction_start = time.perf_counter()
            if prefix is None:
                hidden, context = reduce_hidden(output.hidden_states, mask, self.schema['reduced'])
                confidence = confidence_features(output.logits[0, positions-1], ids[0,positions])
                prefix = combine_features(hidden, confidence, context, self.schema['reduced'])
                del hidden, context, confidence
            sync()
            reduced_seconds = time.perf_counter()-reduction_start
            attention_start = time.perf_counter()
            added, audit = reduce_attentions(output.attentions, mask, self.schema['attention'])
            sync()
            attention_seconds = time.perf_counter()-attention_start
            vector = list(prefix)+added
            if len(vector) != len(self.schema['names']) or not np.isfinite(vector).all():
                raise FloatingPointError('Combined vector incomplete or nonfinite')
            del output, ids, mask, positions
        sync()
        return dict(features=vector, seconds=time.perf_counter()-started, forward_seconds=forward_seconds,
            attention_enabled_forward_seconds=forward_seconds, attention_modules_seconds=module_seconds,
            attention_aggregation_seconds=attention_seconds, reduced_aggregation_seconds=reduced_seconds,
            reduction_seconds=reduced_seconds+attention_seconds, output_shapes=shapes, attention_audit=audit,
            reduced_prefix_source='validated_saved_features' if plan.get('verified_reduced_features') is not None else 'unchanged_reductions_same_forward',
            timing_note='CUDA event attention_modules_seconds includes Q/K/V, RoPE, attention and output projection; not an isolated softmax cost. Attention materialization is included in forward_seconds.',
            cuda_available=torch.cuda.is_available(), gpu_before_allocated_gib=before,
            gpu_after_allocated_gib=torch.cuda.memory_allocated(device)/2**30 if cuda else 0.,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if cuda else 0.,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if cuda else 0.)
