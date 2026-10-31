"""Section 2.1 reconstruction and OUR fusion experiment, not recovered author code.
See AUDIT.md for every mathematical/heuristic assumption. Scoring is GPU-only;
structural weighting can be reused on CPU.
"""
import hashlib
import json
import re
from pathlib import Path
from shared.sersem_baseline import digest, write_json


class StructuralWeights:
    """CPU-only access to the existing SERSEM reconstruction's exact weights."""
    def __init__(self):
        from ASTExtractor import ASTExtractor, PARSERS, AST_AVAILABLE, ENCHANT_AVAILABLE, en_dict
        from LinterExtractor import LinterExtractor
        import shutil
        if not AST_AVAILABLE or not ENCHANT_AVAILABLE or shutil.which('flake8') is None:
            raise RuntimeError('Exact structural weights require AST parsers, en_US dictionary and flake8; no fallback')
        self.parsers, self.dictionary = PARSERS, en_dict
        self.extractor, self.linter = ASTExtractor(), LinterExtractor()
        self.psych = re.compile(r'(TODO|FIXME|HACK|Note to self)', re.IGNORECASE)

    def char_weights(self, text, language):
        import numpy as np
        if language not in self.parsers:
            raise ValueError(f'Unsupported language {language}; no regex-only substitution')
        encoded = text.encode('utf-8')
        root = self.parsers[language].parse(encoded).root_node
        weights = np.full(len(text), .1, dtype=np.float32)
        nodes = [root]
        while nodes:
            node = nodes.pop()
            start = len(encoded[:node.start_byte].decode('utf-8', errors='replace'))
            end = len(encoded[:node.end_byte].decode('utf-8', errors='replace'))
            value = None
            if 'comment' in node.type:
                value = 10.
            elif 'string' in node.type:
                value = 5.  # all literal strings: no undocumented entropy/length cutoff
            elif 'identifier' in node.type:
                ident = text[start:end]
                words = self.extractor.split_identifier(ident)
                unknown = sum(len(w) >= 4 and not self.dictionary.check(w) for w in words)
                multilingual = bool(words) and unknown > 0 and unknown / len(words) >= .5
                if multilingual:
                    value = 10.
                elif len(ident) >= 10:  # Table 1 boundary; prose instead says >10
                    value = 3.
                elif re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{3,}', ident):
                    value = 1.
            if value is not None:
                weights[start:end] = np.maximum(weights[start:end], value)
            nodes.extend(node.children)
        for match in self.psych.finditer(text):
            weights[match.start():match.end()] = 10.
        lines = text.splitlines(keepends=True)
        starts = np.cumsum([0] + [len(line) for line in lines]).tolist()
        for line, column in self.linter.extract_errors(text, language):
            if not (1 <= line <= len(lines) and column >= 1):
                raise ValueError('Invalid linter source position')
            offset = starts[line - 1] + column - 1
            left, right = max(0, offset - 2), min(len(text), offset + 3)
            weights[left:right] = np.maximum(weights[left:right], 5.)
        return weights, bool(root.has_error)


class WeightedAnomalyMIA(StructuralWeights):
    """Grey-box branch: raw next-token logits + language-aware structural weights.

    z_t=(logit[t-1, x_t]-mean_v(logit[t-1,v]))/max(std_v,epsilon)
    S=sum_t w_t*sigmoid(z_t)/sum_t w_t, over non-special, nonempty targets.
    Vocabulary moments are uniform population moments, NOT softmax moments.
    """
    def __init__(self, model, tokenizer, cfg, out):
        import torch
        if not torch.cuda.is_available() or model.device.type != 'cuda':
            raise RuntimeError('WeightedAnomalyMIA requires a Kaggle CUDA model; no CPU path')
        super().__init__()
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.cache = Path(out) / 'greybox_checkpoints'; self.cache.mkdir(exist_ok=True)
        if cfg.get('resources', {}).get('enabled'):
            from shared.resource_pipeline import semantic_cfg
            self.identity = json.dumps(semantic_cfg(cfg), sort_keys=True)
        else:
            self.identity = json.dumps(cfg, sort_keys=True)
        if cfg['greybox']['epsilon'] <= 0 or cfg['greybox']['logit_chunk_size'] < 1:
            raise ValueError('Invalid grey-box numerical configuration')

    def one(self, text, language):
        import torch
        import numpy as np
        key = digest(self.identity + language + digest(text))
        path = self.cache / f'{key}.json'
        checkpoint_store = getattr(self, '_checkpoint_store', None)
        cached = checkpoint_store.get(key) if checkpoint_store is not None else json.loads(path.read_text()) if path.exists() else None
        if cached is not None:
            saved = cached
            if saved.get('key') != key or not np.isfinite(saved['greybox_score']) or not 0 <= saved['greybox_score'] <= 1 or saved['greybox_n_targets'] < 1:
                raise ValueError('Invalid grey-box checkpoint')
            return saved
        char, parse_errors = self.char_weights(text, language)
        enc = self.tokenizer(text, return_tensors='pt', truncation=True,
                             max_length=self.cfg['max_length'], add_special_tokens=True,
                             return_offsets_mapping=True, return_special_tokens_mask=True)
        offsets = enc.pop('offset_mapping')[0].tolist()
        special = enc.pop('special_tokens_mask')[0].tolist()
        n = enc.input_ids.shape[1]
        if n < 2:
            raise ValueError('No causal next-token target after truncation')
        target_weights = []
        for i, (start, end) in enumerate(offsets):
            if not 0 <= start <= end <= len(text):
                raise ValueError('Tokenizer offsets are incompatible with character indexing')
            target_weights.append(float(char[start:end].max()) if i > 0 and not special[i] and start < end else 0.)
        weights = torch.tensor(target_weights[1:], dtype=torch.float32, device=self.model.device)
        if not bool((weights > 0).any()):
            raise ValueError('No eligible next-token targets')
        inputs = enc.to(self.model.device)
        try:
            with torch.no_grad():
                # No activation hooks, hidden states, probes, gradients, or softmax.
                output = self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
                logits = output.logits[0, :-1, :]
                targets = inputs.input_ids[0, 1:]
                numerator = torch.zeros((), dtype=torch.float32, device=logits.device)
                for start in range(0, n - 1, self.cfg['greybox']['logit_chunk_size']):
                    end = min(n - 1, start + self.cfg['greybox']['logit_chunk_size'])
                    block = logits[start:end].float()
                    var, mean = torch.var_mean(block, dim=-1, correction=0)
                    correct = block.gather(1, targets[start:end, None]).squeeze(1)
                    z = (correct - mean) / var.sqrt().clamp_min(self.cfg['greybox']['epsilon'])
                    numerator += (torch.sigmoid(z) * weights[start:end]).sum()
                value = float((numerator / weights.sum()).item())
        except torch.OutOfMemoryError as exc:
            raise RuntimeError('Grey-box GPU OOM; use more VRAM or explicitly configure a new smoke run. No fallback.') from exc
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Nonfinite/out-of-range grey-box score')
        saved = {'key': key, 'greybox_score': value, 'greybox_n_targets': int((weights > 0).sum().item()), 'greybox_ast_has_error': parse_errors}
        if checkpoint_store is None:
            write_json(path, saved)
        else:
            checkpoint_store.put(key, saved)
        return saved

    def score_frame(self, frame):
        import pandas as pd
        rows = []
        for i, row in enumerate(frame.itertuples(index=False)):
            rows.append(self.one(row.content, row.language))
            if (i + 1) % 10 == 0:
                print(f'Grey-box checkpointed {i + 1}/{len(frame)}')
        return pd.DataFrame(rows, index=frame.index).drop(columns=['key'])


def split_calibration(calibration, seed, out):
    """Fusion selection and threshold fitting must use different samples."""
    from sklearn.model_selection import train_test_split
    fusion, threshold = train_test_split(calibration, test_size=.5, random_state=seed,
                                         stratify=calibration.language + calibration.is_member.astype(str))
    fusion, threshold = fusion.reset_index(drop=True), threshold.reset_index(drop=True)
    fusion['split_role'], threshold['split_role'] = 'fusion_fit', 'threshold_calibration'
    if set(fusion.content_sha256) & set(threshold.content_sha256):
        raise ValueError('Calibration partitions overlap')
    for name, frame in [('fusion_fit', fusion), ('threshold_calibration', threshold)]:
        frame.drop(columns=['content'], errors='ignore').to_parquet(out / f'{name}_manifest.parquet', index=False)
    return fusion, threshold


def score_branches(attack, greybox, frame, cfg, path):
    from shared.sersem_baseline import score
    result = frame.copy()
    if set(cfg['branches']) & {'lumia', 'fusion'}:
        result = score(attack, result, path)
    if set(cfg['branches']) & {'greybox', 'fusion'}:
        grey = greybox.score_frame(frame)
        for column in grey:
            result[column] = grey[column]
    result.drop(columns=['content'], errors='ignore').to_parquet(path, index=False)
    return result


def fit_fusion(fusion_fit, cfg, out):
    """OUR convex mixture on raw [0,1] scores; selection uses fusion_fit only."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    setting = cfg['fusion']
    candidates = []
    if 'fusion' not in cfg['branches']:
        state = {'status': 'disabled'}
    else:
        if setting['mode'] == 'fixed':
            alpha = float(setting['alpha'])
        elif setting['mode'] == 'fit_grid':
            if fusion_fit.is_member.nunique() != 2 or not fusion_fit.split_role.eq('fusion_fit').all():
                raise ValueError('Fusion requires separate two-class fusion_fit data')
            for a in setting['alpha_grid']:
                a = float(a)
                if not 0 <= a <= 1:
                    raise ValueError('Fusion grid must be in [0,1]')
                mixed = a * fusion_fit.lumia_score + (1 - a) * fusion_fit.greybox_score
                candidates.append({'alpha': a, 'fusion_fit_auc': float(roc_auc_score(fusion_fit.is_member, mixed))})
            if not candidates:
                raise ValueError('Empty fusion grid')
            # Deterministic tie: closest to prespecified alpha, then smaller alpha.
            best = sorted(candidates, key=lambda x: (-x['fusion_fit_auc'], abs(x['alpha'] - setting['alpha']), x['alpha']))[0]
            alpha = best['alpha']
        else:
            raise ValueError('fusion.mode must be fixed or fit_grid')
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Fusion alpha must be in [0,1]')
        state = {'status': 'our experiment, not author fusion', 'formula': 'alpha*lumia_score+(1-alpha)*greybox_score',
                 'mode': setting['mode'], 'alpha': alpha, 'candidates': candidates,
                 'fit_sample_ids': fusion_fit.sample_id.tolist() if setting['mode'] == 'fit_grid' else [],
                 'fit_content_hashes': fusion_fit.content_sha256.tolist() if setting['mode'] == 'fit_grid' else []}
    write_json(out / 'fusion.json', state)
    return state


def add_fusion(frame, state):
    frame = frame.copy()
    if state['status'] != 'disabled':
        a = state['alpha']
        frame['fusion_score'] = a * frame.lumia_score + (1 - a) * frame.greybox_score
    return frame


def freeze_thresholds(calibration, cfg, out):
    import numpy as np
    from sklearn.metrics import roc_curve
    if not calibration.split_role.eq('threshold_calibration').all():
        raise ValueError('Threshold selection requires its separate calibration partition')
    thresholds = {}
    for branch in cfg['branches']:
        scores = calibration[f'{branch}_score']
        fpr, _, cuts = roc_curve(calibration.is_member, scores, drop_intermediate=False)
        valid = np.where((fpr <= .01) & np.isfinite(cuts))[0]
        cut = float(cuts[valid[-1]]) if len(valid) else float(np.nextafter(scores.max(), np.inf))
        thresholds[branch] = cut
    state = {'source': 'threshold_calibration only', 'target_fpr': .01, 'thresholds': thresholds,
             'sample_ids': calibration.sample_id.tolist(), 'content_hashes': calibration.content_sha256.tolist()}
    write_json(out / 'thresholds.json', state)
    return state


def evaluate_branches(test, cfg, out):
    """Report once with frozen mixture and thresholds; never select on test."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    if (out / 'metrics.json').exists():
        raise RuntimeError('Held-out evaluation already completed for this run')
    if not test.split_role.eq('heldout_test').all():
        raise ValueError('Expected heldout_test samples')
    frozen = json.loads((out / 'thresholds.json').read_text())
    fusion = json.loads((out / 'fusion.json').read_text())
    if frozen['source'] != 'threshold_calibration only':
        raise ValueError('Threshold provenance mismatch')
    forbidden = set(frozen['content_hashes']) | set(fusion.get('fit_content_hashes', []))
    if forbidden & set(test.content_sha256):
        raise ValueError('Held-out overlap with fitting/calibration')
    def metrics(frame, column, threshold):
        y, s = frame.is_member.to_numpy(), frame[column].to_numpy()
        if not np.isfinite(s).all() or not ((s >= 0) & (s <= 1)).all():
            raise ValueError('Invalid membership scores')
        f, t, _ = roc_curve(y, s)
        p = s >= threshold
        return {'n': len(y), 'negatives': int(sum(y == 0)), 'auc_roc': float(roc_auc_score(y, s)),
                'tpr_at_fpr_le_1pct_roc': float(max(t[f <= .01])),
                'calibrated_threshold_tpr': float(p[y == 1].mean()), 'calibrated_threshold_fpr': float(p[y == 0].mean()),
                'low_fpr_resolution_limited': bool(sum(y == 0) < 100)}
    report = {'status': 'manual Kaggle execution; no published-result equivalence claim', 'branches': {}}
    for branch in cfg['branches']:
        col, cut = f'{branch}_score', frozen['thresholds'][branch]
        row = {'provenance': {'lumia': 'unchanged upstream scoring', 'greybox': 'Section 2.1 reconstruction', 'fusion': 'our configurable fusion experiment'}[branch],
               'overall': metrics(test, col, cut), 'by_language': {}}
        for lang, group in test.groupby('language'):
            row['by_language'][lang] = metrics(group, col, cut) if group.is_member.nunique() == 2 and group.is_member.value_counts().min() >= 2 else {'status': 'insufficient samples'}
        report['branches'][branch] = row
    fig, axes = plt.subplots(1, len(cfg['branches']), figsize=(6 * len(cfg['branches']), 4), squeeze=False)
    for ax, branch in zip(axes[0], cfg['branches']):
        for label, group in test.groupby('is_member'):
            ax.hist(group[f'{branch}_score'], bins=30, alpha=.5, density=True, label=f'membership={label}')
        ax.set(title=branch, xlabel='Score', ylabel='Density'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'score_distributions.png', dpi=160); plt.show()
    test.drop(columns=['content'], errors='ignore').to_parquet(out / 'heldout_test_scores.parquet', index=False)
    test.drop(columns=['content', 'lumia_confidence_vector'], errors='ignore').to_csv(out / 'sample_scores.csv', index=False)
    write_json(out / 'metrics.json', report)
    return report
