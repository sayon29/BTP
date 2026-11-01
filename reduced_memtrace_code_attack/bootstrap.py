"""Stdlib-only verified source discovery/extraction, callable before dependency setup."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

MARKER = 'reduced_memtrace_code_attack/source_manifest.json'
ZIP_NAME = 'reduced_memtrace_kaggle_source.zip'


def stage_source(input_root, project_root, expected_sha, upload_source='', upload_zip=''):
    root, project = Path(input_root), Path(project_root)
    markers = sorted(root.rglob('source_manifest.json'))
    candidates = sorted({p.parent.parent for p in markers if p.parent.name=='reduced_memtrace_code_attack'} |
                        {p for p in root.rglob(ZIP_NAME)})
    print('Reduced memTrace source candidates:', [str(p) for p in candidates])
    if not (project/MARKER).is_file():
        source = Path(upload_zip) if upload_zip else Path(upload_source) if upload_source else None
        if source is None:
            if len(candidates) != 1:
                raise FileNotFoundError('Attach reduced_memtrace_kaggle_source.zip or extracted package; set UPLOAD_SOURCE/UPLOAD_ZIP if ambiguous. Candidates: '+str(candidates))
            source = candidates[0]
        if source.is_dir() and not (source/MARKER).is_file():
            source = source/ZIP_NAME
        if project.exists():
            raise FileExistsError('Incomplete PROJECT_ROOT; select a fresh directory')
        def safe(name):
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source path: '+name)
            destination = (project/name).resolve()
            if not destination.is_relative_to(project.resolve()):
                raise ValueError('Unsafe source path: '+name)
            return destination
        if source.is_dir():
            raw = (source/MARKER).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                raise ValueError('Outdated source manifest')
            manifest = json.loads(raw)
            for name in list(manifest)+[MARKER]:
                destination = safe(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, destination)
        else:
            with zipfile.ZipFile(source) as archive:
                if hashlib.sha256(archive.read(MARKER)).hexdigest() != expected_sha:
                    raise ValueError('Outdated ZIP manifest')
                if len(archive.namelist()) != len(set(archive.namelist())):
                    raise ValueError('Duplicate ZIP entries')
                for entry in archive.infolist():
                    safe(entry.filename)
                    if entry.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError('Source ZIP symlinks forbidden')
                archive.extractall(project)
    raw = (project/MARKER).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Source manifest pin mismatch')
    manifest = json.loads(raw)
    for name, expected in manifest.items():
        path = (project/name).resolve()
        if not path.is_relative_to(project.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Source hash mismatch: '+name)
    return manifest
