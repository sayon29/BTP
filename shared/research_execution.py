"""Optional Kaggle research timing/reporting; no change to attack equations.
Import is lightweight. All scientific/model work happens only when called on Kaggle.
"""
import contextlib
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from shared.sersem_baseline import write_json

MILESTONES = (500, 2000, 5000, 25000)


def auc_metrics(frame, column):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    y, s = frame.is_member.to_numpy(), frame[column].to_numpy(dtype=float)
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite score in evaluated cohort')
    if len(set(y)) != 2:
        return {'n': len(y), 'status': 'two classes required'}, set()
    f, t, cuts = roc_curve(y, s, drop_intermediate=False)
    eligible = np.flatnonzero(f <= .01)
    chosen = eligible[np.argmax(t[eligible])]
    cut = cuts[chosen]
    detected = set(frame.loc[(y == 1) & (s >= cut), 'sample_id'])
    return dict(n=len(y), negatives=int(sum(y == 0)), members=int(sum(y == 1)),
                auc_roc=float(roc_auc_score(y, s)), tpr_at_fpr_le_1pct=float(t[chosen]),
                achieved_fpr=float(f[chosen]), diagnostic_roc_threshold=float(cut) if np.isfinite(cut) else None,
                low_fpr_resolution_limited=bool(sum(y == 0) < 100)), detected


def metric_report(frame, columns):
    report = {}
    for column in columns:
        report[column] = {'overall': auc_metrics(frame, column)[0], 'by_language': {}}
        for language, group in frame.groupby('language'):
            report[column]['by_language'][language] = auc_metrics(group, column)[0]
    return report


def install_probe_timer(attack, out):
    """Call the original trainer verbatim, recording completed-layer wall time."""
    if getattr(attack, '_research_probe_timer', False):
        return
    original = attack._train_probe
    def timed(*args, **kwargs):
        import torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        layer = kwargs.get('layer_idx', args[4] if len(args) > 4 else None)
        event = {'layer': layer, 'seconds': time.perf_counter() - start,
                 'includes': 'original probe fit and early-stop validation; excludes layer-selection reporting'}
        with (Path(out) / 'probe_timing.jsonl').open('a') as file:
            file.write(json.dumps(event) + '\n')
        return result
    attack._train_probe = timed
    attack._research_probe_timer = True


class _NoCache:
    def get(self, key):
        return None
    def put(self, key, value):
        pass


def benchmark_sersem(attack, greybox, store, shadow, cfg, out, count=30):
    """20--50 training examples spread across language/length, plus cap stress.
    No probe fitting/heldout labels; caller runs on Kaggle CUDA. Saves measured ETA.
    """
    import numpy as np
    import pandas as pd
    import torch
    from shared.resource_pipeline import safe_extract
    if not 20 <= count <= 50:
        raise ValueError('Benchmark count must be 20..50')
    out = Path(out)
    # Reuse completed benchmark only for this exact model/math and helper source.
    identity = hashlib.sha256((json.dumps(cfg, sort_keys=True) + str(count) + Path(__file__).read_text()).encode()).hexdigest()
    path = out / 'timing_benchmark.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report['identity'] == identity:
            return report
    candidate_parts = []
    for _, group in shadow.groupby('language', sort=True):
        ordered = group.sort_values('content_chars')
        positions = np.unique(np.linspace(0, len(ordered) - 1, min(20, len(ordered))).astype(int))
        candidate_parts.append(ordered.iloc[positions])
    candidates = pd.concat(candidate_parts).drop_duplicates('sample_id')
    metadata = []
    for batch in store.batches(candidates, 1):
        row = batch.iloc[0]
        tokens = len(attack.tokenizer(row.content, truncation=True, max_length=cfg['max_length'], add_special_tokens=True).input_ids)
        metadata.append(dict(sample_id=row.sample_id, tokens=tokens, language=row.language))
    lengths = pd.DataFrame(metadata)
    # Round-robin language/token-length ordering, unique samples, no labels.
    bins = pd.cut(lengths.tokens, [0, 512, 2048, 4096, cfg['max_length'] + 1], duplicates='drop') if cfg['max_length'] > 4096 else pd.cut(lengths.tokens, 4)
    lengths['bin'] = bins.astype(str)
    queues = [g.sort_values('tokens').sample(frac=1, random_state=cfg['seed']).sample_id.tolist()
              for _, g in lengths.groupby(['language', 'bin'], observed=True)]
    selected = []
    limit = min(count,49,len(lengths))
    while len(selected) < limit and any(queues):
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    longest = lengths.sort_values('tokens').iloc[-1].sample_id
    if longest not in selected:
        selected[-1] = longest
    selected = selected[:limit]
    rows = []
    both = set(cfg['branches']) & {'lumia', 'fusion'}
    old_cache = getattr(greybox, '_checkpoint_store', None) if greybox is not None else None
    if greybox is not None:
        greybox._checkpoint_store = _NoCache()  # force actual work; do not overwrite valid score caches
    def measure(text, language, sid, synthetic=False):
        tokens = len(attack.tokenizer(text, truncation=True, max_length=cfg['max_length'], add_special_tokens=True).input_ids)
        record = dict(sample_id=sid, language=language, tokens=tokens, synthetic=synthetic)
        for branch in ['lumia', 'greybox']:
            if branch == 'lumia' and not both or branch == 'greybox' and greybox is None:
                continue
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            if branch == 'lumia':
                values = safe_extract(attack, [text]); del values
            else:
                greybox.one(text, language)
            torch.cuda.synchronize()
            record[branch + '_seconds'] = time.perf_counter() - started
            record[branch + '_peak_gpu_gib'] = torch.cuda.max_memory_allocated() / 2**30
        rows.append(record)
        write_json(out / 'timing_benchmark_partial.json', rows)
    try:
        for batch in store.batches(shadow[shadow.sample_id.isin(selected)], 1):
            row = batch.iloc[0]
            measure(row.content, row.language, row.sample_id)
        if max(row['tokens'] for row in rows) < cfg['max_length']:
            # Explicit synthetic allocation/latency stress, excluded from ETA means.
            unit = '# Timing stress comment; synthetic input, never an evaluation sample.\nx = 12345\n'
            text = unit
            while len(attack.tokenizer(text, add_special_tokens=True).input_ids) < cfg['max_length']:
                text *= 2
            measure(text, 'Python', '__synthetic_full_context_stress__', True)
    finally:
        if greybox is not None:
            if old_cache is None:
                del greybox._checkpoint_store
            else:
                greybox._checkpoint_store = old_cache
    natural = pd.DataFrame([r for r in rows if not r['synthetic']])
    summary = {}
    for branch in ['lumia', 'greybox']:
        col = branch + '_seconds'
        if col in natural:
            summary[branch] = {'mean_seconds': float(natural[col].mean()), 'p90_seconds': float(natural[col].quantile(.9))}
    report = dict(identity=identity, measured_on='user-run Kaggle CUDA only', samples=rows, summary=summary,
                  layers=attack.num_layers, cap_exercised=max(r['tokens'] for r in rows) == cfg['max_length'],
                  note='Length/language-stratified training timing, not a measured evaluation-length distribution. Synthetic stress excluded from mean. No throughput claim until executed.')
    pd.DataFrame(rows).to_csv(out / 'timing_samples.csv', index=False)
    write_json(path, report)
    return report


def estimate_sersem_runtime(cfg, out, shadow_rows, fusion_rows, calibration_rows, heldout_rows):
    """Forecast from measured benchmark and completed probe times only."""
    out = Path(out)
    path = out / 'timing_benchmark.json'
    if not path.exists():
        return {'status': 'run Kaggle benchmark first; no invented throughput'}
    report = json.loads(path.read_text()); summary = report['summary']
    lumia = bool(set(cfg['branches']) & {'lumia', 'fusion'})
    grey = bool(set(cfg['branches']) & {'greybox', 'fusion'})
    groups = math.ceil(report['layers'] / cfg['resources']['layer_group_size'])
    probe_file = out / 'probes' / ("probes_" + cfg['model_id'].replace('/', '_') + '.manifest.json')
    fitting_needed = lumia and cfg['probe_mode'] == 'train' and not probe_file.exists()
    eval_rows = calibration_rows + heldout_rows + (fusion_rows if 'fusion' in cfg['branches'] else 0)
    feature_seconds = groups * shadow_rows * summary['lumia']['mean_seconds'] if fitting_needed else 0
    score_seconds = eval_rows * sum(summary[b]['mean_seconds'] for b, enabled in [('lumia', lumia), ('greybox', grey)] if enabled)
    times = [json.loads(line)['seconds'] for line in (out / 'probe_timing.jsonl').read_text().splitlines()] if (out / 'probe_timing.jsonl').exists() else []
    probe_seconds = sum(times) / len(times) * report['layers'] if times and fitting_needed else 0 if not fitting_needed else None
    result = dict(status='conditional forecast from actual measured samples; not a deadline guarantee',
                  feature_seconds_estimate=feature_seconds, scoring_seconds_estimate=score_seconds,
                  probe_fit_seconds_estimate=probe_seconds, completed_probe_timings=len(times),
                  attack_compute_seconds_estimate=feature_seconds + score_seconds + probe_seconds if probe_seconds is not None else None,
                  additional_unmeasured_components=['source scans/cache I/O', 'model/setup per new session', 'layer selection and output reporting'],
                  note='No full wall-time estimate until all components measured. Training benchmark length mixture may differ from heldout. Recompute after fitting; retained checkpoints reduce remaining work.')
    write_json(out / 'runtime_forecast.json', result)
    print(json.dumps(result, indent=2)); return result


def score_with_milestones(attack, greybox, frame, cfg, path):
    """Instrument original bounded scoring; no altered batching or equations."""
    from shared.resource_pipeline import score_bounded
    if frame.empty or not frame.split_role.eq('heldout_test').all():
        return score_bounded(attack, greybox, frame, cfg, path)
    import pandas as pd
    out = Path(path).parent; db = attack._resource_context['scores']
    from shared.resource_pipeline import semantic_cfg, file_hash
    from shared.sersem_baseline import digest
    expected = set(frame.sample_id); completed = {}
    mathematical_identity=json.dumps(semantic_cfg(cfg),sort_keys=True)
    probe_file=attack.cache_dir / ('probes_'+cfg['model_id'].replace('/','_')+'.pkl')
    probe_hash=file_hash(probe_file) if set(cfg['branches']) & {'lumia','fusion'} else ''
    valid_keys={digest(digest(mathematical_identity+r.sample_id+r.content_sha256+r.split_role)+probe_hash)
                for r in frame.itertuples(index=False)}
    with contextlib.closing(sqlite3.connect(db.path)) as con:
        for key,packed in con.execute('SELECT key,value FROM records ORDER BY rowid'):
            row = json.loads(packed)
            if key in valid_keys and row.get('split_role') == 'heldout_test' and row.get('sample_id') in expected:
                completed[row['sample_id']] = row
    start = time.perf_counter(); initial = len(completed)
    original = db.put
    def snapshot(n):
        rows = list(completed.values())[:n]
        result = pd.DataFrame(rows)
        fusion_path = out / 'fusion.json'
        if fusion_path.exists():
            from shared.weighted_anomaly import add_fusion
            result = add_fusion(result, json.loads(fusion_path.read_text()))
        result.to_parquet(out / f'checkpoint_{n}_scores.parquet', index=False)
        write_json(out / f'checkpoint_{n}_metrics.json', dict(
            diagnostic_only='Do not select attack settings on these heldout reports',
            completed=n, metrics=metric_report(result, [b + '_score' for b in cfg['branches']]),
            observed_current_session_seconds=time.perf_counter() - start))
        elapsed=time.perf_counter()-start;fresh=len(completed)-initial
        write_json(out/'heldout_progress_runtime.json',dict(completed=len(completed),requested=len(frame),
                   current_session_seconds=elapsed,new_examples=fresh,
                   examples_per_second=fresh/elapsed if elapsed else None,
                   remaining_seconds_estimate=(len(frame)-len(completed))*elapsed/fresh if fresh else None,
                   note='Observed heldout scoring+checkpoint throughput; no hidden-label fitting; length/cache mix may change.'))
    for n in MILESTONES:
        if len(completed) >= n and not (out / f'checkpoint_{n}_scores.parquet').exists():
            snapshot(n)
    def put(key, value):
        original(key, value)
        if key in valid_keys and value.get('split_role') == 'heldout_test' and value.get('sample_id') in expected:
            completed[value['sample_id']] = value
            if len(completed) in MILESTONES and not (out / f'checkpoint_{len(completed)}_scores.parquet').exists():
                snapshot(len(completed))
    db.put = put
    try:
        result = score_bounded(attack, greybox, frame, cfg, path)
    finally:
        db.put = original
    elapsed = time.perf_counter() - start
    write_json(out / 'heldout_runtime.json', dict(current_session_seconds=elapsed,
               newly_scored_examples=len(completed) - initial, cached_examples=initial,
               examples_per_second=(len(completed) - initial) / elapsed if elapsed else None,
               note='Measured orchestration/scoring time for this invocation, excludes earlier sessions/model setup.'))
    return result
