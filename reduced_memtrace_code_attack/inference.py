"""One transformer forward per sample; all hidden states, no attention or KV cache."""
import time
from reduced_memtrace_code_attack.features import reduce_hidden, confidence_features, combine_features


class Extractor:
    def __init__(self, model, schema):
        self.model, self.schema = model, schema
        model.eval()
        model.requires_grad_(False)

    def one(self, plan):
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
        with torch.inference_mode():
            ids = torch.tensor([plan['input_ids']], device=device, dtype=torch.long)
            mask = torch.tensor([plan['attention_mask']], device=device, dtype=torch.long)
            positions = torch.tensor(plan['target_positions'], device=device, dtype=torch.long)
            forward_start = time.perf_counter()
            output = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                output_attentions=False, use_cache=False, return_dict=True)
            sync()
            forward_seconds = time.perf_counter()-forward_start
            if getattr(output, 'attentions', None) is not None:
                raise RuntimeError('Unexpected attention matrices materialized; stop and investigate')
            shapes = dict(hidden_states=[list(h.shape) for h in output.hidden_states], logits=list(output.logits.shape))
            reduction_start = time.perf_counter()
            hidden, context = reduce_hidden(output.hidden_states, mask, self.schema)
            confidence = confidence_features(output.logits[0, positions-1], ids[0, positions])
            vector = combine_features(hidden, confidence, context, self.schema)
            sync()
            reduction_seconds = time.perf_counter()-reduction_start
            del output, hidden, context, confidence, ids, mask, positions
        sync()
        return dict(features=vector, seconds=time.perf_counter()-started,
            forward_seconds=forward_seconds, reduction_seconds=reduction_seconds, output_shapes=shapes,
            cuda_available=torch.cuda.is_available(), gpu_before_allocated_gib=before,
            gpu_after_allocated_gib=torch.cuda.memory_allocated(device)/2**30 if cuda else 0.,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if cuda else 0.,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if cuda else 0.)
