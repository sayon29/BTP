"""Resource probe, durable per-sample features, and atomic exports."""
from pathlib import Path
import time
from adaptive_token_attack.data import IDENTITY
from attention_memtrace_code_attack.data import cache_key
from shared.sersem_baseline import write_json


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def probe_rows(shadow, plans):
    """One shadow sample per language at staggered length quantiles; includes longest."""
    result = []
    for i, (_, group) in enumerate(shadow.groupby('language', sort=True)):
        ordered = sorted(group.sample_id, key=lambda sid: (plans[sid]['metadata']['source_token_count'], sid))
        result.append(ordered[round((len(ordered)-1)*i/4)])
    if len(result) != 5:
        raise ValueError('Resource probe requires five shadow languages')
    return result


def run_features(selected, plans, identity, cache, extractor_factory, monitor, out):
    import numpy as np
    import pandas as pd
    out = Path(out)
    rows = pd.concat(list(selected.values()), ignore_index=True)
    records, measurements, extractor = {}, [], None
    started = time.perf_counter()
    def extract(row, probe=False):
        nonlocal extractor
        monitor.guard()
        key = cache_key(row, identity)
        value = cache.get('features', key)
        cached = value
        if value is None or probe:
            if extractor is None:
                extractor = extractor_factory()
            result = extractor.one(plans[row['sample_id']])
            # Fail before accepting an unsafe feature batch. Existing cache remains intact.
            if max(result['gpu_peak_allocated_gib'], result['gpu_peak_reserved_gib']) > identity['memory_stop_gib']:
                write_json(out/'memory_stop.json', result)
                raise RuntimeError('GPU peak exceeds frozen 11 GiB resource guard; preserve cache and investigate')
            value = dict(features=result['features'], metadata=plans[row['sample_id']]['metadata'])
            if cached is None:
                cache.put('features', key, value)
            else:
                # Probe verifies resources on the current session; never overwrites completed vectors.
                value = cached
            measurement = dict(sample_id=row['sample_id'], **monitor.snapshot())
            measurement.update(result)
            measurement.pop('features')
            measurements.append(measurement)
        vector = np.asarray(value['features'], dtype=float)
        if vector.shape != (len(identity['schema']['names']),) or not np.isfinite(vector).all():
            raise ValueError('Invalid cached feature vector')
        records[row['sample_id']] = dict({k:row[k] for k in IDENTITY}, **value['metadata'],
            **dict(zip(identity['schema']['names'], value['features'])))
        return value
    try:
        # Five actual attention forwards on every session with pending samples; existing cache entries stay untouched.
        missing = any(cache.get('features', cache_key(r, identity)) is None for r in rows.to_dict('records'))
        if missing:
            lookup = selected['shadow'].set_index('sample_id', drop=False)
            for sid in probe_rows(selected['shadow'], plans):
                extract(lookup.loc[sid].to_dict(), probe=True)
                write_json(out/'resource_probe.json', dict(samples=measurements,
                    maximum_observed_source_token_length=max(plans[s]['metadata']['source_token_count'] for s in probe_rows(selected['shadow'], plans)),
                    scientific_identity=identity['schema']['sha256'], labels_used=False,
                    attention_feature_count=identity['schema']['attention_features'],
                    combined_feature_count=len(identity['schema']['names']), probe_fresh_forwards=len(measurements)))
            # Exercise the maximum scientific span if real probe inputs were all short.
            probe_ids = probe_rows(selected['shadow'], plans)
            if max(len(plans[s]['input_ids']) for s in probe_ids) < identity['analysis_tokens']:
                longest = max(probe_ids, key=lambda sid: len(plans[sid]['input_ids']))
                tokens = plans[longest]['input_ids']
                cap = identity['analysis_tokens']
                stress = dict(input_ids=(tokens*((cap+len(tokens)-1)//len(tokens)))[:cap],
                    attention_mask=[1]*cap, target_positions=list(range(1,cap)))
                monitor.guard()
                if extractor is None:
                    extractor = extractor_factory()
                measurement = extractor.one(stress)
                measurement.pop('features')
                write_json(out/'resource_stress.json', dict(**monitor.snapshot(), synthetic=True, measurement=measurement))
                if max(measurement['gpu_peak_allocated_gib'], measurement['gpu_peak_reserved_gib']) > identity['memory_stop_gib']:
                    raise RuntimeError('Synthetic 128-position stress exceeds 11 GiB memory guard')
            if len(measurements) != 5:
                raise RuntimeError('Five-sample attention resource probe incomplete')
        for i, row in enumerate(rows.to_dict('records')):
            extract(row)
            elapsed = time.perf_counter()-started
            event = dict(stage='attention_memtrace_features', completed=i+1, total=len(rows),
                seconds_per_sample=elapsed/(i+1), remaining_seconds_estimate=(len(rows)-i-1)*elapsed/(i+1),
                **monitor.snapshot())
            write_json(out/'progress.json', event)
            monitor.tick('attention_memtrace_features', i+1, len(rows))
    finally:
        extractor = None
        atomic_parquet(pd.DataFrame(list(records.values())), out/'features.parquet')
        write_json(out/'extraction_measurements.json', measurements)
    frame = pd.DataFrame(list(records.values()))
    times = [m['seconds'] for m in measurements]
    timing = dict(samples=len(frame), fresh_forwards=len(times), session_seconds=time.perf_counter()-started,
        mean_fresh_seconds=float(np.mean(times)) if times else None,
        estimated_full_3000_extraction_minutes=float(np.mean(times)*3000/60) if times else None,
        mean_attention_aggregation_seconds=float(np.mean([m['attention_aggregation_seconds'] for m in measurements])) if measurements else None,
        gpu_peak_allocated_gib=max((m['gpu_peak_allocated_gib'] for m in measurements), default=None),
        gpu_peak_reserved_gib=max((m['gpu_peak_reserved_gib'] for m in measurements), default=None),
        estimate_note='Measured extraction only; excludes setup, source retrieval, planning and RF fit. No estimate on cache-only runs.')
    write_json(out/'timing.json', timing)
    return frame, timing
