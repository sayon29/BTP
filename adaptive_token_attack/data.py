"""Load complete saved partitions, then optionally select smoke subsets. Never reconstruct splits."""
import hashlib
import json
from pathlib import Path

FILES = {'shadow': 'shadow_manifest.parquet', 'validation': 'fusion_fit_manifest.parquet',
         'calibration': 'threshold_calibration_manifest.parquet', 'heldout': 'heldout_test_manifest.parquet',
         'scores': 'heldout_test_scores.parquet', 'config': 'config.json'}
COUNTS = {'shadow': 1600, 'validation': 200, 'calibration': 200, 'heldout': 1000}
ROLES = {'shadow': {'shadow', 'shadow_train', 'shadow_early_stop', 'shadow_layer_selection'},
         'validation': {'fusion_fit'}, 'calibration': {'threshold_calibration'}, 'heldout': {'heldout_test'}}
IDENTITY = ['sample_id', 'language', 'is_member', 'content_sha256', 'official_split', 'source_row', 'split_role']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def frame_fingerprint(frame):
    return fingerprint(json.loads(frame[IDENTITY].to_json(orient='records')))


def resolve_paths(baseline_run='', overrides=None, input_root='/kaggle/input'):
    """Discover one run, including incomplete runs so missing-file errors are useful."""
    overrides = {k: Path(v) for k, v in (overrides or {}).items() if v}
    if set(overrides) - set(FILES):
        raise ValueError('Unknown manifest path keys: ' + str(set(overrides)-set(FILES)))
    if baseline_run:
        root = Path(baseline_run)
    elif set(overrides) == set(FILES):
        root = None
    else:
        candidates = sorted({p.parent for name in FILES.values() if name != 'config.json'
                             for p in Path(input_root).rglob(name)})
        if len(candidates) > 1:
            raise ValueError('Multiple saved runs found; set BASELINE_RUN explicitly:\n' + '\n'.join(map(str, candidates)))
        root = candidates[0] if candidates else Path(input_root) / '<select-saved-SERSEM-run>'
    paths = {k: overrides[k] if k in overrides else root / name for k, name in FILES.items()}
    missing = [f'{k}: {p}' for k, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing saved SERSEM files (no splits will be regenerated):\n' + '\n'.join(missing))
    return paths


def validate_partition(frame, role):
    required = set(IDENTITY) | {'membership'}
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError(f'{role}: missing/null identity columns')
    if (frame.sample_id.duplicated().any() or frame.content_sha256.duplicated().any()
            or not frame.split_role.isin(ROLES[role]).all()):
        raise ValueError(f'{role}: duplicate identity/content or incorrect split_role')
    if not frame.membership.map({'member': 1, 'non-member': 0}).eq(frame.is_member).all():
        raise ValueError(f'{role}: membership labels disagree')
    if not frame.official_split.eq('test' if role == 'heldout' else 'train').all():
        raise ValueError(f'{role}: official split violation')
    if frame.is_member.nunique() != 2:
        raise ValueError(f'{role}: both membership classes required')
    if not frame.content_sha256.str.fullmatch(r'[0-9a-f]{64}').all():
        raise ValueError(f'{role}: invalid content hashes')
    if not frame.source_row.map(lambda x: isinstance(x, (int, float)) and x >= 0 and int(x) == x).all():
        raise ValueError(f'{role}: invalid source rows')
    expected = frame.language + '/' + frame.official_split + '/' + frame.source_row.astype(int).astype(str)
    if not frame.sample_id.eq(expected).all():
        raise ValueError(f'{role}: sample ID/source position disagreement')


def validate_partitions(partitions, expected_counts=None):
    expected_counts = COUNTS if expected_counts is None else expected_counts
    if set(partitions) != set(COUNTS):
        raise ValueError('Require shadow, validation, calibration and heldout partitions')
    ids, hashes = set(), set()
    for role, frame in partitions.items():
        validate_partition(frame, role)
        if len(frame) != expected_counts[role]:
            raise ValueError(f'{role}: expected {expected_counts[role]} saved rows, got {len(frame)}')
        if ids & set(frame.sample_id) or hashes & set(frame.content_sha256):
            raise ValueError(f'{role}: partition overlap in IDs or content; data leakage')
        ids.update(frame.sample_id); hashes.update(frame.content_sha256)


def match_scores(manifest, scores):
    import numpy as np
    columns = ['lumia_score', 'greybox_score', 'fusion_score']
    fields = ['sample_id', 'language', 'is_member', 'content_sha256']
    if not set(fields + columns) <= set(scores) or scores.sample_id.duplicated().any():
        raise ValueError('Saved scores require unique IDs, identity fields and all three branches')
    joined = manifest[fields].merge(scores[fields+columns], on='sample_id', how='left',
                                    suffixes=('', '_saved'), validate='one_to_one')
    for key in fields[1:]:
        if not joined[key].eq(joined[key+'_saved']).all():
            raise ValueError('Saved score identity mismatch/missing sample: ' + key)
    if not np.isfinite(joined[columns].to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite saved SERSEM scores')
    return joined[fields+columns]


def load_run(paths):
    import pandas as pd
    partitions = {role: pd.read_parquet(paths[role]) for role in COUNTS}
    validate_partitions(partitions)
    scores = match_scores(partitions['heldout'], pd.read_parquet(paths['scores']))
    config = json.loads(Path(paths['config']).read_text())
    for key in ['model_id', 'model_revision', 'dataset_id', 'dataset_revision', 'dtype', 'max_length', 'seed', 'greybox', 'versions', 'upstream_sha']:
        if key not in config:
            raise ValueError('Missing saved baseline configuration field: ' + key)
    return partitions, scores, config


def seal(payload):
    return dict(payload, artifact_sha256=fingerprint(payload))


def check_seal(artifact):
    payload = {k:v for k,v in artifact.items() if k != 'artifact_sha256'}
    if artifact.get('artifact_sha256') != fingerprint(payload):
        raise ValueError('Artifact checksum mismatch')


SMOKE_SIZES = dict(shadow=80, validation=20, calibration=20, heldout=30)


def select_partitions(original, mode, smoke_sizes=None, seed=17):
    """Validate complete saved partitions FIRST; select rows within each partition.

    Smoke quotas balance membership exactly within each language, with language
    pair counts differing by at most one. Seeded SHA256 ranks avoid dependence on
    input row ordering, Python hash randomization or a split-generation routine.
    """
    validate_partitions(original)  # never accept truncated originals, in either mode
    if mode not in {'smoke', 'full'}:
        raise ValueError('MODE must be smoke or full')
    sizes = dict(COUNTS) if mode == 'full' else dict(SMOKE_SIZES if smoke_sizes is None else smoke_sizes)
    if set(sizes) != set(COUNTS):
        raise ValueError('SMOKE_SIZES must specify shadow, validation, calibration and heldout')

    def rank(*parts):
        return hashlib.sha256(json.dumps([int(seed), *parts], separators=(',', ':')).encode()).hexdigest()

    selected = {}
    for role, frame in original.items():
        if mode == 'full':
            selected[role] = frame.copy()
            continue
        size = sizes[role]
        languages = sorted(frame.language.unique())
        if (type(size) is not int or size % 2 or size < 2*len(languages) or size > len(frame)):
            raise ValueError(f'{role}: smoke count must be even, at least {2*len(languages)} (one pair per language), and <= {len(frame)}')
        language_order = sorted(languages, key=lambda lang: (rank(role, lang), lang))
        base, extra = divmod(size//2, len(languages))
        ids = []
        for index, language in enumerate(language_order):
            pairs = base + int(index < extra)
            for label in [0, 1]:
                group = frame[frame.language.eq(language) & frame.is_member.eq(label)]
                if len(group) < pairs:
                    raise ValueError(f'{role}: insufficient saved {language}/{label} rows for balanced smoke quota {pairs}')
                ordered = sorted(group.sample_id, key=lambda sid: (rank(role, language, label, sid), sid))
                ids.extend(ordered[:pairs])
        selected[role] = frame.set_index('sample_id', drop=False).loc[sorted(ids)].reset_index(drop=True).copy()
        # Explicit identity check, including membership text, without changing source positions.
        fields = IDENTITY + ['membership']
        reference = frame.set_index('sample_id').loc[selected[role].sample_id, fields[1:]].reset_index()
        if not selected[role][fields].equals(reference[fields]):
            raise ValueError('Subset selection changed saved identity: '+role)
    validate_partitions(selected, expected_counts=sizes)
    execution = seal(dict(schema='adaptive-selection-v1', mode=mode, seed=int(seed), expected_counts=sizes,
                          selection_policy='all-saved-rows-v1' if mode == 'full' else 'seeded-hash-language-membership-v1',
                          original_manifests={k:frame_fingerprint(v) for k,v in original.items()},
                          selected_manifests={k:frame_fingerprint(v) for k,v in selected.items()}))
    return selected, execution


def validate_execution(execution, partitions, complete=True):
    """Bind stage inputs to the validated selection without changing scoring math."""
    check_seal(execution)
    if execution['mode'] not in {'smoke', 'full'} or set(execution['expected_counts']) != set(COUNTS):
        raise ValueError('Invalid execution mode/counts')
    if execution['mode'] == 'full' and execution['expected_counts'] != COUNTS:
        raise ValueError('Full mode requires production counts 1600/200/200/1000')
    if complete:
        validate_partitions(partitions, execution['expected_counts'])
    for role, frame in partitions.items():
        validate_partition(frame, role)
        if len(frame) != execution['expected_counts'][role] or frame_fingerprint(frame) != execution['selected_manifests'][role]:
            raise ValueError('Execution selection/count/identity mismatch: '+role)


def execution_labels(execution):
    mode = execution['mode'] if execution else 'full'
    return dict(mode=mode, preliminary=mode == 'smoke', resolution_limited=mode == 'smoke')
