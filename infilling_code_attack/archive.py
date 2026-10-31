"""Validate and archive completed Kaggle results without touching the cache."""
from pathlib import Path
import shutil
import sqlite3
import zipfile


REQUIRED_OUTPUTS = (
    'config.json', 'metrics.json', 'infilling_scores.parquet',
    'matched_comparison.parquet', 'run_manifest.json',
    'reproducibility.json', 'infilling_cache.sqlite',
)


def create_results_archive(out, work_root, run_id):
    """Create a ZIP of OUT after the caller has closed its SQLite writer."""
    out, work_root = Path(out), Path(work_root)
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('RUN_ID must be a plain directory name')
    if not out.is_dir():
        raise FileNotFoundError(f'Results directory does not exist: {out}')
    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError('Cannot archive incomplete results; missing: ' + ', '.join(missing))
    database = out / 'infilling_cache.sqlite'
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
    target = work_root / f'{run_id}.zip'
    generated = Path(shutil.make_archive(str(target.with_suffix('')), 'zip', root_dir=out))
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
