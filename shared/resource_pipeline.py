"""Bounded Stage 1 orchestration. Original upstream math stays in its checkout.
All runtime entry points below are executed only by the user on Kaggle GPU.
"""
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import sqlite3
import time
from pathlib import Path
from shared.sersem_baseline import digest, write_json


def semantic_cfg(cfg):
    """Allow relocation/resume budgets without changing mathematical identity."""
    value = json.loads(json.dumps(cfg))
    settings = value.get('resources', {})
    for key in ['model_cache_dir', 'model_local_files_only', 'data_access', 'disk_reserve_gib',
                'ram_reserve_gib', 'text_cache_gib', 'log_interval_seconds', 'session_budget_minutes']:
        settings.pop(key, None)
    return value


class ResourcePause(RuntimeError):
    """A safe between-chunk/layer stop; completed work is already durable locally."""


class Monitor:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, Path(out)
        self.started = time.monotonic()
        self.stage, self.last_time, self.last_count = None, self.started, 0

    def snapshot(self):
        import psutil
        import torch
        disk = shutil.disk_usage(self.out)
        result = {'ram_total_gib': psutil.virtual_memory().total / 2**30,
                  'disk_used_gib': disk.used / 2**30, 'rss_gib': psutil.Process().memory_info().rss / 2**30,
                  'ram_available_gib': psutil.virtual_memory().available / 2**30,
                  'disk_free_gib': disk.free / 2**30}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(gpu_name=torch.cuda.get_device_name(0), gpu_free_gib=free / 2**30, gpu_total_gib=total / 2**30,
                          gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                          gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                          gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        return result

    def tick(self, stage, done, total=None, force=False):
        now = time.monotonic()
        if self.stage != stage:
            self.stage, self.last_time, self.last_count = stage, now, done
        if not force and now - self.last_time < self.cfg['resources']['log_interval_seconds']:
            return
        elapsed = now - self.last_time
        rate = (done - self.last_count) / elapsed if elapsed > 0 else 0
        event = dict(stage=stage, completed=int(done), total=total,
                     elapsed_session_seconds=now - self.started, samples_per_second=rate,
                     remaining_seconds_estimate=(total - done) / rate if total and rate > 0 else None,
                     **self.snapshot())
        with (self.out / 'resource_log.jsonl').open('a') as file:
            file.write(json.dumps(event) + '\n')
        print(json.dumps(event))
        self.last_time, self.last_count = now, done

    def guard(self, needed_bytes=0):
        import psutil
        setting = self.cfg['resources']
        if shutil.disk_usage(self.out).free < needed_bytes + setting['disk_reserve_gib'] * 2**30:
            raise ResourcePause('Insufficient free disk for the next allocation plus reserve. Checkpoints kept; persist them before ending the session. No cache was auto-deleted.')
        if psutil.virtual_memory().available < setting['ram_reserve_gib'] * 2**30:
            raise ResourcePause('Available system RAM is below reserve; checkpoints kept. Resume on adequate resources.')
        if setting['session_budget_minutes'] and time.monotonic() - self.started > 60 * setting['session_budget_minutes']:
            raise ResourcePause('Configured session budget reached at a checkpoint boundary. Save/persist the run directory, then resume. A partially trained probe restarts from its prior boundary.')


def restore_run(source, out):
    """Copy a trusted saved run from read-only Kaggle input, without overwrites."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        return  # same-session restart already uses the writable run directory
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Saved run and destination must be separate directories')
    if not (source / 'config.json').is_file():
        raise FileNotFoundError('RESUME_FROM must contain a trusted run config.json')
    out.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('Resume directories must contain real files, not external symlinks')
        if not path.is_file():
            continue
        target = out / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != file_hash(path):
                raise FileExistsError(f'Resume would overwrite a different file: {target}')
            continue
        if shutil.disk_usage(out).free < path.stat().st_size + 2**30:
            raise ResourcePause('Not enough space to restore checkpoints while preserving 1 GiB reserve')
        # An interrupted restore must never leave a partial final checkpoint.
        temporary = target.with_name(target.name + '.restore.tmp')
        shutil.copy2(path, temporary)
        if file_hash(temporary) != file_hash(path):
            raise ValueError(f'Restored file checksum mismatch: {path}')
        temporary.replace(target)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cached_weights(cfg):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    settings = cfg['resources']
    args = dict(repo_id=cfg['model_id'], revision=cfg['model_revision'],
                cache_dir=settings['model_cache_dir'], local_files_only=True)
    for index_name, single_name in [('model.safetensors.index.json', 'model.safetensors'),
                                    ('pytorch_model.bin.index.json', 'pytorch_model.bin')]:
        try:
            index = Path(hf_hub_download(filename=index_name, **args))
        except LocalEntryNotFoundError:
            try:
                single = Path(hf_hub_download(filename=single_name, **args))
                return [single]
            except LocalEntryNotFoundError:
                continue
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        try:
            return [Path(hf_hub_download(filename=name, **args)) for name in names]
        except LocalEntryNotFoundError:
            return []  # partial index/shards are not a complete usable snapshot
    return []


def directory_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def preflight(cfg, out, monitor, shadow_rows=None):
    """Config/metadata and capacity only; no model instantiated or inferred."""
    from transformers import AutoConfig
    from datasets import config as datasets_config
    setting = cfg['resources']
    if setting['faithful'] and (cfg['max_length'] != 8192 or cfg['dtype'] != 'float16' or setting['feature_dtype'] != 'float32' or setting['attention_backend'] != 'upstream'):
        raise ValueError('Faithful profile requires 8192-token truncation, reference fp16 model, and float32 pooled-feature storage and upstream attention selection. Select faithful=False explicitly for approximations.')
    if setting['feature_dtype'] not in {'float32', 'float16'} or setting['layer_group_size'] < 1:
        raise ValueError('Invalid feature dtype/group size')
    try:
        model_cfg = AutoConfig.from_pretrained(cfg['model_id'], revision=cfg['model_revision'],
                                              cache_dir=setting['model_cache_dir'],
                                              local_files_only=setting['model_local_files_only'])
    except OSError as exc:
        raise RuntimeError('Pinned model config is missing from MODEL_CACHE_DIR; attach the complete previously used HF cache/snapshot or explicitly allow Kaggle downloads with adequate disk.') from exc
    n = int(shadow_rows) if shadow_rows is not None else int(5 * cfg['dev_per_language'] * .8)
    layers, width, vocab = model_cfg.num_hidden_layers, 2 * model_cfg.hidden_size, model_cfg.vocab_size
    feature_item_bytes = 4 if setting['feature_dtype'] == 'float32' else 2
    training_needed = bool(set(cfg['branches']) & {'lumia', 'fusion'}) and cfg['probe_mode'] == 'train'
    model_tag = cfg['model_id'].replace('/', '_')
    if (Path(out) / 'probes' / f'probes_{model_tag}.manifest.json').is_file():
        training_needed = False  # prepare_probes still verifies this completed artifact
    group = min(layers, setting['layer_group_size']) if training_needed else 0
    per_layer = n * width * feature_item_bytes
    files = cached_weights(cfg)
    cached = sum(p.stat().st_size for p in files)
    free = shutil.disk_usage(out).free
    if setting['model_local_files_only'] and not files:
        raise RuntimeError('Complete model weights are not present at MODEL_CACHE_DIR for this exact revision. Attach/cache them separately; no automatic download is allowed by this run configuration.')
    download = 0
    if not files:
        from huggingface_hub import HfApi
        metadata = HfApi().model_info(cfg['model_id'], revision=cfg['model_revision'], files_metadata=True)
        weights = [f for f in metadata.siblings if '/' not in f.rfilename and f.rfilename.endswith('.safetensors')]
        if not weights:
            weights = [f for f in metadata.siblings if '/' not in f.rfilename and f.rfilename.startswith('pytorch_model') and f.rfilename.endswith('.bin')]
        if not weights or any(f.size is None for f in weights):
            raise RuntimeError('Cannot establish weight download size; attach a complete local cache instead.')
        download = sum(f.size for f in weights)
    existing_group = directory_bytes(Path(out) / 'active_features')
    remaining_text = max(0, setting['text_cache_gib'] * 2**30 - directory_bytes(Path(out) / 'selected_text.sqlite'))
    remaining_index = max(0, .35 * 2**30 - directory_bytes(Path(out) / 'source_index.sqlite'))
    required_new = (max(0, per_layer * group - existing_group) + remaining_text + remaining_index
                    + int((setting['disk_reserve_gib'] + .20) * 2**30) + download)
    notes = ['Peak attention/workspace memory is backend-dependent; the longest-sample GPU validation is required.',
             'Layer grouping trades feature disk/RAM for repeated source scans and target-model forwards.',
             'Model/cache sizes are physical-file estimates, not a guarantee of free GPU memory.']
    if cfg['max_length'] >= 8192:
        notes.append('8192 tokens on T4 may OOM even with bounded pooled features; no context/precision fallback occurs.')
    # Original trainer still copies a full layer's fit/validation rows. Include
    # mapped pages, indexing/tensor staging and bounded all-layer pooled chunks.
    host_working = (per_layer * group + n * width * 4 * 1.8
                    + cfg['feature_chunk_size'] * layers * width * 4 * 2)
    snapshot = monitor.snapshot()
    if host_working / 2**30 + setting['ram_reserve_gib'] > snapshot['ram_available_gib']:
        notes.append('Estimated host working set plus RAM reserve exceeds available RAM. Reduce layer grouping or use more RAM; original per-layer trainer is not minibatch-streamed.')
    report = {'model_layers': layers, 'pooled_width': width, 'vocabulary_size': vocab,
              'shadow_rows': n, 'all_layers_feature_gib': n * layers * width * 4 / 2**30,
              'active_group_feature_gib': per_layer * group / 2**30,
              'layer_groups_and_source_passes': math.ceil(layers / group) if group else 0,
              'cached_weight_gib': cached / 2**30,
              'model_cache_physical_gib': directory_bytes(setting['model_cache_dir']) / 2**30,
              'run_directory_gib': directory_bytes(out) / 2**30,
              'active_feature_cache_gib': existing_group / 2**30,
              'source_index_gib': directory_bytes(Path(out) / 'source_index.sqlite') / 2**30,
              'dataset_cache_physical_gib': directory_bytes(datasets_config.HF_DATASETS_CACHE) / 2**30,
              'selected_text_cache_gib': directory_bytes(Path(out) / 'selected_text.sqlite') / 2**30,
              'checkpoint_gib': directory_bytes(Path(out) / 'probe_training.pkl') / 2**30,
              'final_probes_and_provenance_gib': directory_bytes(Path(out) / 'probes') / 2**30,
              'score_store_gib': directory_bytes(Path(out) / 'scores.sqlite') / 2**30,
              'logits_fp16_gib_per_forward': cfg['max_length'] * vocab * 2 / 2**30,
              'logits_fp32_gib_per_forward': cfg['max_length'] * vocab * 4 / 2**30,
              'weighted_activation_float32_gib_per_layer': cfg['max_length'] * model_cfg.hidden_size * 4 / 2**30,
              'one_layer_probe_fit_and_val_gpu_gib': n * width * 4 * .8 / 2**30,
              'estimated_additional_host_working_gib': host_working / 2**30,
              'new_disk_allowance_gib': required_new / 2**30, 'notes': notes, **snapshot}
    write_json(Path(out) / 'resource_preflight.json', report)
    print(json.dumps(report, indent=2))
    if free < required_new:
        raise ResourcePause(f'Preflight needs about {required_new / 2**30:.2f} GiB more free disk including reserves; only {free / 2**30:.2f} GiB available. Reduce LAYER_GROUP_SIZE/TEXT_CACHE_GIB explicitly, or attach model caches/add storage. No user cache is auto-deleted.')
    monitor.guard()
    return report


def safe_extract(attack, texts):
    """Call unchanged upstream extractor per text so its final logits die promptly."""
    import numpy as np
    result = {i: [] for i in range(attack.num_layers)}
    for text in texts:
        log = io.StringIO()
        modules = attack._get_layer_modules()
        before = {id(m): set(m._forward_hooks) for m in modules}
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                values = attack._upstream_extract([text])
        finally:
            for m in modules:
                for key in set(m._forward_hooks) - before[id(m)]:
                    del m._forward_hooks[key]
        if 'Error extracting activations' in log.getvalue():
            raise RuntimeError('Upstream extraction failed, possibly GPU OOM; no zero-feature fallback accepted. ' + log.getvalue()[-2000:])
        for i in result:
            x = values[i]
            if x.shape != (1, 2 * attack.model.config.hidden_size) or not np.isfinite(x).all():
                raise RuntimeError('Invalid upstream pooled activations')
            result[i].append(x)
    return {i: np.vstack(rows) for i, rows in result.items()}


def rng_state():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def atomic_pickle(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {'payload': payload, 'sha256': hashlib.sha256(payload).hexdigest()}
    with tmp.open('wb') as file:
        pickle.dump(envelope, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush(); os.fsync(file.fileno())
    tmp.replace(path)


def train_grouped(attack, cfg, shadow, out):
    """Exact upstream _train_probe, split and selection; bounded layer features.
    No incremental/online probe fitting and no changes to early-stop snapshot.
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from LumiaAttack import LinearProbe
    context = attack._resource_context
    store, monitor = context['store'], context['monitor']
    path = Path(out) / 'probe_training.pkl'
    signature = digest(json.dumps(semantic_cfg(cfg), sort_keys=True) + json.dumps(shadow.content_sha256.tolist()))
    if path.exists():
        with path.open('rb') as file:
            envelope = pickle.load(file)  # trusted own-run checkpoint only
        if hashlib.sha256(envelope['payload']).hexdigest() != envelope['sha256']:
            raise ValueError('Probe checkpoint checksum mismatch')
        state = pickle.loads(envelope['payload'])
        if state['signature'] != signature:
            raise ValueError('Probe checkpoint configuration/training order mismatch')
        attack.probes.clear()
        for i, weights in state['probes'].items():
            probe = LinearProbe(2 * attack.model.config.hidden_size, attack.hidden_dim).to(attack.model.device)
            probe.load_state_dict(weights); probe.eval(); attack.probes[i] = probe
        restore_rng(state['rng'])
    else:
        labels = shadow.is_member.to_numpy()
        indices = [[], [], []]
        for label in [0, 1]:
            idx = np.where(labels == label)[0]; np.random.shuffle(idx)
            n, v = int(len(idx) * .60), int(len(idx) * .20)
            for destination, values in zip(indices, [idx[:n], idx[n:n + v], idx[n + v:]]):
                destination.extend(values)
        state = dict(signature=signature, indices=[np.array(x) for x in indices], probes={}, aucs={}, rng=rng_state())
        atomic_pickle(path, state)  # durable split/RNG before the first feature pass
    train, val, select = state['indices']
    attack.train_indices, attack.val_indices, attack.test_indices = train, val, select
    attack.activation_dim = 2 * attack.model.config.hidden_size
    labels = shadow.is_member.to_numpy()
    root = Path(out) / 'active_features'; root.mkdir(exist_ok=True)
    setting = cfg['resources']; group_size = setting['layer_group_size']
    for start in range(0, attack.num_layers, group_size):
        group = list(range(start, min(start + group_size, attack.num_layers)))
        group_dir = root / f'group_{start:03d}'
        if all(i in state['probes'] for i in group):
            if group_dir.exists():
                shutil.rmtree(group_dir)  # only committed, reproducible temporary features
            continue
        group_dir.mkdir(exist_ok=True)
        marker = group_dir / 'identity.json'
        identity = dict(signature=signature, layers=group, rows=len(shadow), width=attack.activation_dim, dtype=setting['feature_dtype'])
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError('Active feature-group identity mismatch')
        write_json(marker, identity)
        arrays = {}
        missing_before = [i for i in group if not (group_dir / f'layer_{i}.npy').exists()]
        for layer in group:
            file = group_dir / f'layer_{layer}.npy'
            if file.exists():
                array = np.load(file, mmap_mode='r+')
                if array.shape != (len(shadow), attack.activation_dim) or str(array.dtype) != setting['feature_dtype']:
                    raise ValueError('Feature map shape/dtype mismatch')
            else:
                monitor.guard(len(shadow) * attack.activation_dim * np.dtype(setting['feature_dtype']).itemsize + 4096)
                array = np.lib.format.open_memmap(file, mode='w+', dtype=setting['feature_dtype'], shape=(len(shadow), attack.activation_dim))
            arrays[layer] = array
        flag_path = group_dir / 'complete.npy'
        if flag_path.exists():
            flags = np.load(flag_path, mmap_mode='r+')
            if missing_before and bool(flags.any()):
                raise ValueError('Resume snapshot lost feature files for completed rows; restore the entire active_features directory')
        else:
            flags = np.lib.format.open_memmap(flag_path, mode='w+', dtype='bool', shape=(len(shadow),))
            flags[:] = False; flags.flush()
        if flags.shape != (len(shadow),):
            raise ValueError('Feature completion map mismatch')
        remaining = shadow.loc[~np.asarray(flags)].copy()
        # Preserve positions in ORIGINAL shadow order despite source streaming.
        positions = {sid: i for i, sid in enumerate(shadow.sample_id)}
        for batch in store.batches(remaining, cfg['feature_chunk_size']):
            monitor.guard()
            values = safe_extract(attack, batch.content.tolist())
            pos = np.array([positions[sid] for sid in batch.sample_id])
            for layer in group:
                arrays[layer][pos] = values[layer]; arrays[layer].flush()
            flags[pos] = True; flags.flush()
            del values
            monitor.tick(f'features:layers{group}', int(flags.sum()), len(shadow))
        if not bool(flags.all()):
            raise RuntimeError('Incomplete feature group')
        # Any source/cache machinery must not perturb the probe RNG trajectory.
        restore_rng(state['rng'])
        torch.cuda.empty_cache()
        for layer in group:
            if layer in state['probes']:
                continue
            monitor.guard()
            X = arrays[layer]
            probe, _ = attack._train_probe(X[train], labels[train], X[val], labels[val], layer)
            attack.probes[layer] = probe
            probe.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X[select]).to(attack.model.device)
                predictions = probe(test_tensor).cpu().numpy()
            auc = float(roc_auc_score(labels[select], predictions))
            del test_tensor, predictions
            state['probes'][layer] = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            state['aucs'][layer], state['rng'] = auc, rng_state()
            # Old+new probe checkpoint temporary copies are included in reserves.
            monitor.guard(128 * 1024**2)
            atomic_pickle(path, state)
            monitor.tick('probe_layers', len(state['probes']), attack.num_layers, force=True)
        del X
        flags.flush(); flags._mmap.close(); del flags
        for array in arrays.values():
            array.flush(); array._mmap.close()
        arrays.clear(); del array
        shutil.rmtree(group_dir)
        gc.collect(); torch.cuda.empty_cache()
    sorted_layers = sorted(state['aucs'].items(), key=lambda x: x[1], reverse=True)
    attack.best_layer = max(state['aucs'].items(), key=lambda x: x[1])[0]
    attack.ensemble_layers = [i for i, _ in sorted_layers[:attack.ensemble_size]]
    attack._save_probes()
    return train, val, select


class ScoreStore:
    def __init__(self, path):
        self.path = Path(path)
        with contextlib.closing(sqlite3.connect(self.path)) as con:
            con.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT)'); con.commit()

    def get(self, key):
        with contextlib.closing(sqlite3.connect(self.path)) as con:
            row = con.execute('SELECT value FROM records WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with contextlib.closing(sqlite3.connect(self.path)) as con:
            con.execute('INSERT OR REPLACE INTO records VALUES(?,?)', (key, json.dumps(value))); con.commit()


def score_bounded(attack, greybox, frame, cfg, path):
    """Only pooled features for one score chunk; retain scalar results, not text."""
    import numpy as np
    import pandas as pd
    context = attack._resource_context
    store, monitor, db = context['store'], context['monitor'], context['scores']
    identity = json.dumps(semantic_cfg(cfg), sort_keys=True)
    keys = {row.sample_id: digest(identity + row.sample_id + row.content_sha256 + row.split_role)
            for row in frame.itertuples(index=False)}
    # LUMIA score records also bind to the actual completed probe file checksum.
    probe_file = attack.cache_dir / f"probes_{cfg['model_id'].replace('/', '_')}.pkl"
    probe_hash = file_hash(probe_file) if set(cfg['branches']) & {'lumia', 'fusion'} else ''
    keys = {sid: digest(key + probe_hash) for sid, key in keys.items()}
    records = {sid: db.get(key) for sid, key in keys.items()}
    remaining = frame[frame.sample_id.map(lambda sid: records[sid] is None)]
    for batch in store.batches(remaining, cfg['resources']['score_chunk_size']):
        monitor.guard()
        scored = batch.drop(columns=['content', 'manifest_position']).copy()
        if set(cfg['branches']) & {'lumia', 'fusion'}:
            if not attack.probes or not set(attack.ensemble_layers) <= set(attack.probes):
                raise RuntimeError('Missing original probes; no random fallback')
            scores, vectors = attack.compute_scores_with_vectors(batch.content.tolist(), batch.language.tolist())
            if not np.isfinite(scores).all():
                raise ValueError('Invalid original LUMIA scores')
            scored['lumia_score'], scored['lumia_confidence_vector'] = scores, vectors
        if set(cfg['branches']) & {'greybox', 'fusion'}:
            for column, values in greybox.score_frame(batch).items():
                scored[column] = values.to_numpy()
        for row in scored.to_dict(orient='records'):
            sid = row['sample_id']
            db.put(keys[sid], row); records[sid] = row
        monitor.tick(f'score:{frame.split_role.iloc[0]}', sum(x is not None for x in records.values()), len(frame))
    if any(value is None for value in records.values()):
        raise ValueError('Scoring incomplete; resume from committed chunks')
    result = pd.DataFrame([records[sid] for sid in frame.sample_id])
    if result.sample_id.tolist() != frame.sample_id.tolist():
        raise ValueError('Score manifest order mismatch')
    result.to_parquet(path, index=False)
    return result


def attach_runtime(attack, cfg, out, store, monitor, greybox=None):
    scores = ScoreStore(Path(out) / 'scores.sqlite')
    attack._resource_context = dict(store=store, monitor=monitor, scores=scores)
    if greybox is not None:
        greybox._checkpoint_store = scores
    parameter_bytes = sum(p.numel() * p.element_size() for p in attack.model.parameters())
    write_json(Path(out) / 'loaded_model_resources.json',
               {'parameter_gib': parameter_bytes / 2**30, 'device': str(attack.model.device), **monitor.snapshot()})
    monitor.tick('model_loaded', 0, force=True)
    return attack


def validate_resources(attack, greybox, shadow, cfg):
    """Manual GPU validation on longest TRAIN texts; no held-out tuning."""
    import torch
    context = attack._resource_context
    monitor, store = context['monitor'], context['store']
    chosen = shadow.nlargest(cfg['resources']['validation_examples'], 'content_chars')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); completed = 0; token_lengths = []
    for batch in store.batches(chosen, 1):
        monitor.guard()
        text = batch.content.iloc[0]
        token_lengths.append(len(attack.tokenizer(text, truncation=True, max_length=cfg['max_length'], add_special_tokens=True).input_ids))
        if set(cfg['branches']) & {'lumia', 'fusion'}:
            values = safe_extract(attack, [text]); del values
        if greybox is not None:
            greybox.one(text, batch.language.iloc[0])
        completed += 1
        monitor.tick('longest_training_resource_validation', completed, len(chosen), force=True)
    report = dict(samples=completed, elapsed_seconds=time.monotonic() - started,
                  effective_token_cap=cfg['max_length'], maximum_exercised_tokens=max(token_lengths),
                  exercised_full_token_cap=bool(max(token_lengths) == cfg['max_length']),
                  tokens_per_second=sum(token_lengths) / max(time.monotonic() - started, 1e-9), longest_selected_chars=int(chosen.content_chars.max()),
                  note='Actual selected training texts, not proof of worst-case 8192-token coverage. No labels used to choose context/precision.',
                  **monitor.snapshot())
    write_json(monitor.out / 'resource_validation.json', report)
    return report
