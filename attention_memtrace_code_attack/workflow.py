"""One combined experiment, with completed smoke evidence before full evaluation."""
import hashlib
import json
from pathlib import Path
from adaptive_token_attack.data import fingerprint, seal, check_seal
from shared.sersem_baseline import write_json


def require_completed_smoke(identity, cache_source, out):
    proof_path = Path(out)/'smoke_validation.json'
    if proof_path.is_file():
        proof = json.loads(proof_path.read_text())
        check_seal(proof)
        if proof['scientific_sha256'] != fingerprint(identity):
            raise ValueError('Saved smoke validation uses a different combined scientific identity')
        return proof
    if not cache_source:
        raise ValueError('Run the 150-sample smoke experiment first. Set CACHE_REUSE_FROM to its feature_cache.sqlite for the single full experiment.')
    root = Path(cache_source).parent
    config, metrics, manifest = [json.loads((root/name).read_text()) for name in ['config.json','metrics.json','run_manifest.json']]
    if (config['mode'] != 'smoke' or fingerprint(config['scoring']) != fingerprint(identity)
            or manifest['status'] != 'complete' or metrics['status'] != 'complete'
            or metrics['coverage']['valid_matched'] != 30
            or manifest['selected_counts'] != identity['smoke_counts']
            or metrics['feature_counts']['combined'] != len(identity['schema']['names'])):
        raise ValueError('Full run requires a completed compatible 150-sample combined smoke run')
    proof = seal(dict(scientific_sha256=fingerprint(identity),smoke_complete=True,
        smoke_artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ['config.json','metrics.json','run_manifest.json']}))
    write_json(proof_path,proof)
    return proof
