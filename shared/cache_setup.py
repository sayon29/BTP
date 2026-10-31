"""Explicit first-session asset preparation, called only by manual Kaggle runs.
No model construction/inference; does not change any attack configuration/math.
"""
import json
import re
import shutil
import struct
import zipfile
from pathlib import Path
from shared.sersem_baseline import LANGUAGES, write_json


def _download_allowed_here():
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Automatic asset downloads/preparation are restricted to Kaggle. No local download was attempted.')


def _disk_guard(path, required_bytes, reserve_gib):
    path=Path(path)
    while not path.exists(): path=path.parent
    free=shutil.disk_usage(path).free
    required=int(required_bytes+reserve_gib*2**30)
    if free<required:
        raise RuntimeError(f'Insufficient disk at {path}: {free/2**30:.2f} GiB free; need {required_bytes/2**30:.2f} GiB allocation plus {reserve_gib:.2f} GiB reserve. Choose streaming/attach assets/add space explicitly; no caches were deleted.')
    return free


def _local_file(repo_id, revision, cache_dir, name):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(hf_hub_download(repo_id=repo_id,revision=revision,cache_dir=str(cache_dir),filename=name,local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def _weight_files(index):
    names=sorted(set(index['weight_map'].values()))
    if not names or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise ValueError('Unsafe/empty model shard index')
    return names


def _safetensor_header(path):
    # Validate tensor extents without loading weights or importing torch.
    with path.open('rb') as file:
        prefix=file.read(8)
        if len(prefix)!=8: raise ValueError(f'Truncated safetensors header: {path.name}')
        length=struct.unpack('<Q',prefix)[0]
        if length>100*1024**2 or length<2: raise ValueError('Invalid safetensors header size')
        header=json.loads(file.read(length))
    tensors={name:value for name,value in header.items() if name!='__metadata__'}
    if not tensors or max(value['data_offsets'][1] for value in tensors.values())!=path.stat().st_size-8-length:
        raise ValueError(f'Truncated/malformed safetensors data: {path.name}')
    return set(tensors)


def _verify_model(repo_id, revision, cache_dir, required, sizes=None):
    paths={name:_local_file(repo_id,revision,cache_dir,name) for name in required}
    missing=[name for name,path in paths.items() if path is None or not path.is_file() or path.stat().st_size==0]
    if missing: return None,missing
    for name,path in paths.items():
        if sizes and sizes.get(name) is not None and path.stat().st_size!=sizes[name]:
            raise ValueError(f'Cached file size differs from pinned repository metadata: {name}')
    json.loads(paths['config.json'].read_text())
    if 'tokenizer.json' in paths: json.loads(paths['tokenizer.json'].read_text())
    for name,path in paths.items():
        if name.endswith('.index.json'):
            index=json.loads(path.read_text())
            for shard in _weight_files(index):
                if shard not in paths: raise ValueError('Model shard omitted from verification: '+shard)
                if shard.endswith('.safetensors'):
                    tensors=_safetensor_header(paths[shard])
                    if not {key for key,value in index['weight_map'].items() if value==shard}<=tensors:
                        raise ValueError('Shard index/tensor header mismatch: '+shard)
        elif name.endswith('.safetensors'):
            _safetensor_header(path)
        elif name.endswith('.bin'):
            # Standard Torch state-dict ZIP envelope; no unpickling/tensor load.
            if not zipfile.is_zipfile(path):
                raise ValueError(f'Cannot verify legacy/truncated binary weights: {name}. Use the safetensors snapshot.')
            with zipfile.ZipFile(path) as archive:
                entries=archive.infolist()
                if not any(item.filename.endswith('data.pkl') for item in entries) or any(item.header_offset+item.compress_size>path.stat().st_size for item in entries):
                    raise ValueError('Incomplete Torch weight archive: '+name)
    # Validate actual offline config/tokenizer resolution; no target constructed.
    from transformers import AutoConfig,AutoTokenizer
    AutoConfig.from_pretrained(repo_id,revision=revision,cache_dir=str(cache_dir),local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(repo_id,revision=revision,cache_dir=str(cache_dir),local_files_only=True,use_fast=True)
    if not tokenizer.is_fast: raise ValueError('Required fast tokenizer cannot be loaded offline')
    return paths,[]


def _ensure_model_snapshot(model_id, revision, cache_dir, auto_download, reserve_gib, work_root, report_path):
    """Reuse complete pinned snapshot; otherwise download only necessary files.
    MODEL_LOCAL_FILES_ONLY may stay True: this preparation precedes offline loading.
    """
    if not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Resolve MODEL_REVISION to a full immutable SHA before cache preparation')
    if reserve_gib<3: raise ValueError('Model setup must retain at least 3 GiB free')
    cache_dir=Path(cache_dir)
    config=_local_file(model_id,revision,cache_dir,'config.json')
    token_config=_local_file(model_id,revision,cache_dir,'tokenizer_config.json')
    token_json=_local_file(model_id,revision,cache_dir,'tokenizer.json')
    vocab=_local_file(model_id,revision,cache_dir,'vocab.json');merges=_local_file(model_id,revision,cache_dir,'merges.txt')
    required=['config.json','tokenizer_config.json']
    tokenizer_ready=token_config is not None and (token_json is not None or vocab is not None and merges is not None)
    required+=['tokenizer.json'] if token_json else ['vocab.json','merges.txt']
    weights=[]
    for index_name,single in [('model.safetensors.index.json','model.safetensors'),('pytorch_model.bin.index.json','pytorch_model.bin')]:
        index=_local_file(model_id,revision,cache_dir,index_name)
        if index:
            weights=[index_name]+_weight_files(json.loads(index.read_text()));break
        if _local_file(model_id,revision,cache_dir,single): weights=[single];break
    report=dict(model_id=model_id,revision=revision,cache_dir=str(cache_dir),status='checking',reserve_gib=reserve_gib)
    write_json(report_path,report)
    if config and tokenizer_ready and weights:
        paths,missing=_verify_model(model_id,revision,cache_dir,required+weights)
        if not missing:
            _disk_guard(work_root,0,reserve_gib)
            report.update(status='ready',reused_complete_cache=True,files={n:p.stat().st_size for n,p in paths.items()})
            write_json(report_path,report);print('Reusing verified pinned model snapshot:',revision)
            return report
    if not auto_download:
        raise RuntimeError('Pinned model snapshot is incomplete. Enable AUTO_DOWNLOAD_MODEL_IF_MISSING on Kaggle, or attach its complete cache. MODEL_LOCAL_FILES_ONLY need not be disabled.')
    _download_allowed_here()
    from huggingface_hub import HfApi,hf_hub_download
    info=HfApi().model_info(model_id,revision=revision,files_metadata=True)
    if info.sha!=revision: raise ValueError('Hub returned another model revision')
    files={item.rfilename:item.size for item in info.siblings}
    if 'config.json' not in files or 'tokenizer_config.json' not in files:
        raise ValueError('Pinned model lacks required config/tokenizer configuration')
    tokenizer_files=[name for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json'] if name in files]
    if 'tokenizer.json' not in tokenizer_files and not {'vocab.json','merges.txt'}<=set(tokenizer_files):
        raise ValueError('Pinned model has no supported fast tokenizer assets')
    required=['config.json']+tokenizer_files+(['generation_config.json'] if 'generation_config.json' in files else [])
    index_name=next((name for name in ['model.safetensors.index.json','pytorch_model.bin.index.json'] if name in files),None)
    # Prefer safetensors, including a single safetensors file over a bin index.
    if 'model.safetensors.index.json' not in files and 'model.safetensors' in files:
        index_name=None;weights=['model.safetensors']
    elif index_name:
        _disk_guard(cache_dir,4*1024**2,reserve_gib);_disk_guard(work_root,4*1024**2,reserve_gib)
        index=hf_hub_download(repo_id=model_id,revision=revision,filename=index_name,cache_dir=str(cache_dir))
        weights=[index_name]+_weight_files(json.loads(Path(index).read_text()))
    elif 'pytorch_model.bin' in files: weights=['pytorch_model.bin']
    else: raise ValueError('No supported model weight file/index at pinned revision')
    required+=weights
    if any(name not in files or files[name] is None for name in required):
        raise RuntimeError('Cannot establish exact required asset sizes from Hub metadata; attach a complete cache instead')
    missing=[name for name in required if _local_file(model_id,revision,cache_dir,name) is None]
    remaining=sum(files[name] for name in missing)
    report.update(status='downloading',required_files=required,missing_files=missing,missing_bytes=remaining,
                  download_headroom_bytes=256*1024**2)
    write_json(report_path,report)
    _disk_guard(cache_dir,remaining+256*1024**2,reserve_gib)
    _disk_guard(work_root,remaining+256*1024**2,reserve_gib)
    print(f'Preparing {model_id}@{revision}: {remaining/2**30:.2f} GiB missing; preserving {reserve_gib} GiB plus transfer headroom')
    try:
        for name in missing:
            _disk_guard(cache_dir,files[name]+64*1024**2,reserve_gib)
            hf_hub_download(repo_id=model_id,revision=revision,filename=name,cache_dir=str(cache_dir))
        paths,missing=_verify_model(model_id,revision,cache_dir,required,files)
        if missing: raise RuntimeError('Snapshot still incomplete: '+str(missing))
        _disk_guard(work_root,0,reserve_gib)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='incomplete_or_interrupted',error=str(error))
        write_json(report_path,report)
        raise RuntimeError('Pinned model preparation failed/interrupted; model loading was not started. Inspect model_cache_setup.json, check Internet/disk and rerun this cell. Completed Hub files are reused; partial files are retained for Hub resume.') from error
    report.update(status='ready',reused_complete_cache=False,files={n:p.stat().st_size for n,p in paths.items()})
    write_json(report_path,report)
    return report


def ensure_model_snapshot(model_id, revision, cache_dir, auto_download, reserve_gib, work_root, report_path):
    """Report all cache/metadata/download failures before any model loading."""
    try:
        return _ensure_model_snapshot(model_id,revision,cache_dir,auto_download,reserve_gib,work_root,report_path)
    except (Exception,KeyboardInterrupt) as error:
        report=json.loads(Path(report_path).read_text()) if Path(report_path).exists() else dict(model_id=model_id,revision=revision)
        report.update(status='not_ready',error=str(error) or type(error).__name__)
        write_json(report_path,report)
        raise RuntimeError(f'Model cache setup stopped: {error or type(error).__name__}. See {report_path}; no target was loaded. Completed files are retained; rerun after correcting the cause.') from error


def _ensure_dataset_cache(cfg, auto_prepare, preparation_budget_gib, reserve_gib, work_root, report_path):
    """Explicit cached Arrow preparation or streaming; never auto-switch modes."""
    from datasets import load_dataset_builder,DownloadConfig,DownloadMode,config as dataset_config
    mode=cfg['resources']['data_access']
    revision=cfg['dataset_revision']
    if not re.fullmatch(r'[0-9a-f]{40}',revision): raise ValueError('Dataset revision must be an immutable SHA')
    if mode not in {'cached','streaming'}: raise ValueError('Choose cached or streaming explicitly')
    if reserve_gib<3 or preparation_budget_gib<=0: raise ValueError('Dataset reserve >=3 GiB and positive preparation budget required')
    report=dict(dataset_id=cfg['dataset_id'],revision=revision,mode=mode,cache_dir=str(dataset_config.HF_DATASETS_CACHE),languages={})
    if mode=='streaming':
        report.update(status='streaming_selected',note='No Arrow preparation/full dataset download. SourceStore validates pinned configs/splits/content while iterating. Repeated remote scans require Internet.')
        _disk_guard(work_root,0,reserve_gib);write_json(report_path,report);return report
    missing=[];builders={}
    def inspect(builder):
        paths=[]
        for split in ['train','test']:
            data=builder.as_dataset(split=split)
            names=[item['filename'] for item in data.cache_files]
            if not names or not all(Path(name).is_file() and Path(name).stat().st_size>0 for name in names):
                raise FileNotFoundError('Required Arrow files missing: '+split)
            if not {'content','membership'}<=set(data.column_names): raise ValueError('Unexpected source schema')
            paths.extend(names)
            del data
        return paths
    for language in LANGUAGES:
        try:
            builder=load_dataset_builder(cfg['dataset_id'],language,revision=revision,download_config=DownloadConfig(local_files_only=True))
            paths=inspect(builder)
            report['languages'][language]=dict(status='cached',files=paths)
        except (OSError,ValueError) as error:
            missing.append(language);report['languages'][language]=dict(status='missing_or_incomplete',detail=str(error)[:400])
    write_json(report_path,report)
    if not missing:
        _disk_guard(work_root,0,reserve_gib)
        report['status']='ready';write_json(report_path,report);print('Reusing all pinned Arrow language/split caches');return report
    if not auto_prepare:
        raise RuntimeError(f'Missing pinned Arrow caches for {missing}. Enable AUTO_PREPARE_DATASET_IF_MISSING, or explicitly select DATA_ACCESS="streaming". No mode was changed.')
    _download_allowed_here()
    for language in missing:
        builders[language]=load_dataset_builder(cfg['dataset_id'],language,revision=revision)
    download_sizes=[b.info.download_size for b in builders.values()]
    dataset_sizes=[b.info.dataset_size for b in builders.values()]
    known=all(isinstance(n,int) and n>0 for n in download_sizes+dataset_sizes)
    metadata_peak=sum(download_sizes)+sum(dataset_sizes)+max(dataset_sizes) if known else 0
    allowance=max(preparation_budget_gib*2**30,metadata_peak)
    report.update(status='preparing',missing_languages=missing,metadata_peak_estimate_bytes=metadata_peak if known else None,
                  preparation_allowance_bytes=allowance,reserve_gib=reserve_gib,
                  note='Conservative metadata/budget estimate, not exact compressed-to-Arrow expansion. No alternate source/split is substituted.')
    write_json(report_path,report)
    _disk_guard(dataset_config.HF_DATASETS_CACHE,allowance,reserve_gib);_disk_guard(work_root,allowance,reserve_gib)
    try:
        for language,builder in builders.items():
            _disk_guard(work_root,0,reserve_gib)
            builder.download_and_prepare(download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)
            report['languages'][language]=dict(status='prepared',files=inspect(builder))
            _disk_guard(work_root,0,reserve_gib);write_json(report_path,report)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='incomplete_or_interrupted',error=str(error));write_json(report_path,report)
        raise RuntimeError('Dataset preparation failed/interrupted. Completed language caches are reused on rerun. Check dataset_cache_setup.json and free disk, or explicitly choose streaming; no automatic switch occurred.') from error
    report['status']='ready';write_json(report_path,report);return report


def ensure_dataset_cache(cfg, auto_prepare, preparation_budget_gib, reserve_gib, work_root, report_path):
    """Report preparation failures, without changing selected dataset access mode."""
    try:
        return _ensure_dataset_cache(cfg,auto_prepare,preparation_budget_gib,reserve_gib,work_root,report_path)
    except (Exception,KeyboardInterrupt) as error:
        report=json.loads(Path(report_path).read_text()) if Path(report_path).exists() else dict(dataset_id=cfg['dataset_id'],revision=cfg['dataset_revision'])
        report.update(status='not_ready',error=str(error) or type(error).__name__)
        write_json(report_path,report)
        raise RuntimeError(f'Dataset cache setup stopped: {error or type(error).__name__}. See {report_path}. Existing complete caches are retained; rerun after correcting the cause or explicitly select streaming.') from error
