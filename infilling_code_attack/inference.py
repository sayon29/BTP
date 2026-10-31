"""Native causal StarCoder2 queries; transient logits reduced to sufficient statistics."""
import math


class TorchScorer:
    def __init__(self,model,pad_id=0,head_chunk_rows=32,min_sigma=1e-6):
        self.model=model;model.eval();model.requires_grad_(False)
        self.pad_id=int(pad_id);self.head_chunk_rows=int(head_chunk_rows);self.min_sigma=float(min_sigma)
        self.forward_calls=0;self.head_calls=0

    def synchronize(self):
        import torch
        if self.model.device.type=='cuda':torch.cuda.synchronize(self.model.device)

    def predict_batch(self,requests):
        """Each request: complete IDs (incl last label) and real target positions.

        Right padding affects neither real causal context nor positional indexing.
        Only the final needed label is omitted from the forward's input.
        """
        import torch
        if not requests:raise ValueError('Empty query batch')
        lengths=[len(r['ids'])-1 for r in requests];maximum=max(lengths)
        if min(lengths)<1:raise ValueError('No nonempty causal context')
        with torch.inference_mode():
            inputs=torch.full((len(requests),maximum),self.pad_id,dtype=torch.long,device=self.model.device)
            attention=torch.zeros_like(inputs)
            for b,r in enumerate(requests):
                inputs[b,:lengths[b]]=torch.tensor(r['ids'][:-1],dtype=torch.long,device=self.model.device)
                attention[b,:lengths[b]]=1
            positions=torch.arange(maximum,device=self.model.device)[None,:].expand(len(requests),-1)
            self.forward_calls+=1
            output=self.model.model(input_ids=inputs,attention_mask=attention,position_ids=positions,
                use_cache=False,output_hidden_states=False,return_dict=True)
            locations=[]
            for b,r in enumerate(requests):
                for t in r['positions']:
                    if not 1<=t<len(r['ids']):raise ValueError('Causal label outside real sequence')
                    locations.append((b,t))
            result=[{} for _ in requests]
            for start in range(0,len(locations),self.head_chunk_rows):
                block=locations[start:start+self.head_chunk_rows]
                b=torch.tensor([x[0] for x in block],device=self.model.device)
                t=torch.tensor([x[1]-1 for x in block],device=self.model.device)
                hidden=output.last_hidden_state[b,t,:]
                self.head_calls+=1
                logits=self.model.lm_head(hidden).float()
                lp=torch.log_softmax(logits,dim=-1)
                if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite target model log probabilities')
                logp=lp.double();prob=logp.exp()
                mu=(prob*logp).sum(-1)
                variance=(prob*(logp-mu[:,None]).square()).sum(-1)
                sigma=variance.clamp_min(0).sqrt()
                greedy=lp.argmax(-1)
                for k,(batch,target) in enumerate(block):
                    true_id=requests[batch]['ids'][target]
                    result[batch][str(target)]=dict(log_true=float(lp[k,true_id].item()),
                        log_greedy=float(lp[k,greedy[k]].item()),greedy_id=int(greedy[k].item()),
                        mu=float(mu[k].item()),sigma=float(sigma[k].item()),sigma_used=max(float(sigma[k].item()),self.min_sigma))
            return result
