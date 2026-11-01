"""Validate and archive completed Kaggle results without touching the cache."""
from pathlib import Path
import shutil
import sqlite3
import zipfile


REQUIRED_OUTPUTS = (
    'config.json', 'metrics.json', 'heldout_memtrace_scores.parquet',
    'features.parquet', 'feature_schema.json', 'classifier.joblib', 'threshold.json', 'source_version_hashes.json',
    'matched_comparison.parquet', 'run_manifest.json',
    'reproducibility.json', 'feature_cache.sqlite', 'environment.json',
)


def create_results_archive(out, work_root, run_id, disk_reserve_gib=3.0):
    """Create a ZIP of OUT after the caller has closed its SQLite writer."""
    out, work_root = Path(out), Path(work_root)
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('RUN_ID must be a plain directory name')
    if not out.is_dir():
        raise FileNotFoundError(f'Results directory does not exist: {out}')
    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError('Cannot archive incomplete results; missing: ' + ', '.join(missing))
    database = out / 'feature_cache.sqlite'
    for suffix in ('-journal', '-wal', '-shm'):
        sidecar = Path(str(database) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise RuntimeError(f'SQLite cache has an active/nonempty sidecar: {sidecar}')
    uri = database.resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as connection:
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite cache failed quick_check; refusing to archive')
        if connection.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'delete':
            raise RuntimeError('Expected checkpointed DELETE-mode SQLite cache before archiving')
    work_root.mkdir(parents=True, exist_ok=True)
    if work_root.resolve().is_relative_to(out.resolve()):
        raise ValueError('Archive destination must be outside the results directory')
    files = [p for p in out.iterdir() if p.is_file() and p.suffix in {'.json', '.jsonl', '.parquet', '.sqlite', '.joblib'}]
    needed = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(work_root).free < needed + disk_reserve_gib * 2**30:
        raise RuntimeError('Insufficient disk for results archive plus reserve; results retained')
    target = work_root / f'{run_id}.zip'
    generated = target
    with zipfile.ZipFile(generated, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError('Artifact symlinks forbidden')
            archive.write(path, path.name)
    try:
        with zipfile.ZipFile(generated) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Results ZIP failed CRC validation')
            names = set(archive.namelist())
            missing_archive = set(REQUIRED_OUTPUTS) - names
            if missing_archive:
                raise RuntimeError('Results ZIP is missing required outputs: ' + ', '.join(sorted(missing_archive)))
        return generated
    except Exception:
        if generated.exists():
            generated.unlink()
        raise


def create_partial_archive(out, work_root, run_id, disk_reserve_gib=3.):
    """Preserve interrupted features/configuration without claiming completion."""
    out, work_root = Path(out), Path(work_root)
    if not run_id or Path(run_id).name != run_id or run_id in {'.','..'}:
        raise ValueError('RUN_ID must be a plain directory name')
    if work_root.resolve().is_relative_to(out.resolve()):
        raise ValueError('Archive must be outside results')
    database = out/'feature_cache.sqlite'
    if database.exists():
        with sqlite3.connect(database) as connection:
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            connection.execute('PRAGMA journal_mode=DELETE')
            if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Invalid partial cache')
    files = [p for p in out.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl','.parquet','.sqlite','.joblib'}]
    if any(p.is_symlink() for p in files):
        raise ValueError('Artifact symlinks forbidden')
    if shutil.disk_usage(work_root).free < sum(p.stat().st_size for p in files)+disk_reserve_gib*2**30:
        raise RuntimeError('Insufficient disk; partial outputs retained')
    target = work_root/(run_id+'-partial.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Partial archive CRC failure')
    return target
