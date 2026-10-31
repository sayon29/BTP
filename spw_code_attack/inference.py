"""Frozen StarCoder2 decoder and head, exact-prefix cache including terminal states.

A cached terminal state permits a new true-token probability via the LM head
without re-evaluating the transformer on an equivalent prefix.
"""
import base64
import zlib
import numpy as np


def encode_hidden(array):
    arr=np.asarray(array)
    if arr.dtype not in (np.dtype('float16'),np.dtype('float32')):
        raise ValueError('Only lossless float16/float32 terminal states are supported')
    return base64.b64encode(zlib.compress(arr.tobytes())).decode('ascii')


def decode_hidden(value, width, dtype="float16"):
    arr=np.frombuffer(zlib.decompress(base64.b64decode(value)),dtype=np.dtype(dtype)).copy()
    if arr.shape!=(width,) or not np.isfinite(arr).all():
        raise ValueError('Invalid cached terminal state')
    return arr


class TorchPredictor:
    def __init__(self, model):
        self.model=model
        model.eval();model.requires_grad_(False)
        self.forward_calls=0;self.head_calls=0

    def synchronize(self):
        import torch
        if self.model.device.type=='cuda': torch.cuda.synchronize(self.model.device)

    def _project(self, hidden, targets):
        import torch
        records=[]
        # A fixed one-row head projection is also used for cache extensions.
        # It avoids changing the probability convention when target IDs are added.
        for i,tokens in enumerate(targets):
            self.head_calls+=1
            logits=self.model.lm_head(hidden[i:i+1]).float()[0]
            logp=torch.log_softmax(logits,dim=-1)
            if not torch.isfinite(logp).all(): raise FloatingPointError('Nonfinite model distribution')
            greedy=int(logp.argmax().item())
            terminal=hidden[i].detach().cpu().numpy()
            records.append(dict(greedy_id=greedy,log_greedy=float(logp[greedy].item()),
                log_true={str(t):float(logp[t].item()) for t in sorted(set(tokens))},
                hidden=encode_hidden(terminal),hidden_width=int(hidden.shape[-1]),hidden_dtype=str(terminal.dtype)))
        return records

    def predict_batch(self, prefixes, targets):
        import torch
        if not prefixes or len({len(p) for p in prefixes})!=1 or len(prefixes)!=len(targets):
            raise ValueError('Only nonempty, equal-length, unpadded batches are allowed')
        if not len(prefixes[0]):raise ValueError('Empty prefix is not defined')
        with torch.inference_mode():
            inputs=torch.tensor(prefixes,device=self.model.device,dtype=torch.long)
            self.forward_calls+=1
            output=self.model.model(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                                    use_cache=False,return_dict=True,output_hidden_states=False)
            return self._project(output.last_hidden_state[:,-1,:],targets)

    def extend(self, record, targets):
        import torch
        with torch.inference_mode():
            hidden=torch.tensor(decode_hidden(record['hidden'],record['hidden_width'],record['hidden_dtype']),
                                device=self.model.device,dtype=self.model.dtype)[None,:]
            extra=self._project(hidden,[targets])[0]
        # Keep the canonical existing values; extend only absent target probabilities.
        result=dict(record,log_true=dict(record['log_true']))
        result['log_true'].update({k:v for k,v in extra['log_true'].items() if k not in result['log_true']})
        return result

    def original(self, ids, positions, chunk_size=256):
        """Teacher forcing predicts all original-prefix targets in chunked forwards."""
        import torch
        by_prefix={};past=None;wanted=set(positions)
        with torch.inference_mode():
            for start in range(0,max(positions),chunk_size):
                end=min(start+chunk_size,max(positions))
                inputs=torch.tensor([ids[start:end]],device=self.model.device,dtype=torch.long)
                self.forward_calls+=1
                output=self.model.model(input_ids=inputs,
                    attention_mask=torch.ones((1,end),device=self.model.device,dtype=torch.long),
                    past_key_values=past,use_cache=True,return_dict=True,output_hidden_states=False)
                past=output.past_key_values
                selected=sorted(t for t in wanted if start <= t-1 < end)
                if selected:
                    indexes=torch.tensor([t-1-start for t in selected],device=self.model.device)
                    hidden=output.last_hidden_state[0,indexes,:]
                    records=self._project(hidden,[[ids[t]] for t in selected])
                    by_prefix.update(zip(selected,records))
        return by_prefix
