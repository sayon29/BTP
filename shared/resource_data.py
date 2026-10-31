"""Bounded source ingestion and stable metadata manifests. No local execution.
Reads existing Arrow caches lazily, or explicitly selected HF streaming mode.
"""
import contextlib
import csv
import json
import sqlite3
import zlib
from pathlib import Path
from shared.sersem_baseline import LANGUAGES, digest, write_json


class SourceStore:
    def __init__(self, cfg, out, monitor):
        self.cfg, self.out, self.monitor = cfg, Path(out), monitor
        self.db = self.out / 'source_index.sqlite'
        # This optional, bounded source-text cache is ephemeral and separate
        # from durable checkpoint data. Never delete another run's cache.
        self.text_db = self.out / 'selected_text.sqlite'

    def source(self, language, split):
        from datasets import load_dataset, load_dataset_builder, DownloadConfig
        mode = self.cfg['resources']['data_access']
        if mode not in {'cached', 'streaming'}:
            raise ValueError('data_access must be cached or streaming')
        try:
            if mode == 'cached':
                builder = load_dataset_builder(self.cfg['dataset_id'], language, revision=self.cfg['dataset_revision'],
                                               download_config=DownloadConfig(local_files_only=True))
                # as_dataset reads existing memory-mapped Arrow files only.
                # Never call download_and_prepare: no second Arrow/data copy.
                return builder.as_dataset(split=split)
            return load_dataset(self.cfg['dataset_id'], language, split=split,
                                revision=self.cfg['dataset_revision'], streaming=True)
        except Exception as exc:
            raise RuntimeError(f'{language}/{split}: {mode} source unavailable. cached mode requires existing HF Arrow/download caches; '
                               'select streaming explicitly for remote iteration without a full data download.') from exc

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db)
        conn.execute('PRAGMA cache_size=-16384')  # 16 MiB SQLite page cache
        conn.execute('PRAGMA temp_store=FILE')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def index(self):
        """First pass stores hashes/IDs/labels only, never all source strings."""
        from datasets import get_dataset_config_names, get_dataset_split_names
        configs = get_dataset_config_names(self.cfg['dataset_id'], revision=self.cfg['dataset_revision'])
        if not set(LANGUAGES) <= set(configs):
            raise ValueError(f'Unexpected language configs: {configs}')
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS rows (seq INTEGER PRIMARY KEY, sample_id TEXT UNIQUE, language TEXT, official_split TEXT, source_row INTEGER, is_member INTEGER, membership TEXT, content_sha256 TEXT, content_chars INTEGER)')
            con.execute('CREATE TABLE IF NOT EXISTS excluded (sample_id TEXT PRIMARY KEY, language TEXT, official_split TEXT, source_row INTEGER, membership TEXT, reason TEXT, content_type TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS complete (language TEXT, official_split TEXT, metadata TEXT, PRIMARY KEY(language,official_split))')
            con.execute('CREATE INDEX IF NOT EXISTS hash_split ON rows(content_sha256,official_split)')
            con.execute('CREATE INDEX IF NOT EXISTS strata ON rows(language,official_split,is_member)')
        for language in LANGUAGES:
            splits = get_dataset_split_names(self.cfg['dataset_id'], language, revision=self.cfg['dataset_revision'])
            if not {'train', 'test'} <= set(splits):
                raise ValueError(f'Unexpected splits: {language}: {splits}')
            for split in ['train', 'test']:
                with self.connect() as con:
                    if con.execute('SELECT 1 FROM complete WHERE language=? AND official_split=?', (language, split)).fetchone():
                        continue
                    # A partial split is rescanned deterministically after interruption.
                    con.execute('DELETE FROM rows WHERE language=? AND official_split=?', (language, split))
                    con.execute('DELETE FROM excluded WHERE language=? AND official_split=?', (language, split))
                ds = self.source(language, split)
                counts = {'source_rows': 0, 'eligible_content_rows': 0, 'reasons': {}, 'labels_before': {}, 'labels_after': {}, 'features': str(ds.features)}
                batch, bad = [], []
                with self.connect() as con:
                    for i, row in enumerate(ds):
                        if not {'content', 'membership'} <= set(row):
                            raise ValueError(f'Unexpected fields at {language}/{split}/{i}')
                        label = row['membership']
                        if label not in {'member', 'non-member'}:
                            raise ValueError(f'Unknown membership at {language}/{split}/{i}: {label!r}')
                        if 'language' in row and row['language'] != language:
                            raise ValueError('Language field disagrees with config')
                        text = row['content']
                        reason = ('null_content' if text is None else 'non_string_content' if not isinstance(text, str)
                                  else 'empty_or_whitespace_content' if not text.strip() else None)
                        sid = f'{language}/{split}/{i}'
                        counts['source_rows'] += 1
                        counts['labels_before'][label] = counts['labels_before'].get(label, 0) + 1
                        if reason:
                            bad.append((sid, language, split, i, label, reason, type(text).__name__))
                            counts['reasons'][reason] = counts['reasons'].get(reason, 0) + 1
                        else:
                            batch.append((sid, language, split, i, int(label == 'member'), label, digest(text), len(text)))
                            counts['eligible_content_rows'] += 1
                            counts['labels_after'][label] = counts['labels_after'].get(label, 0) + 1
                        if len(batch) + len(bad) >= 1024:
                            con.executemany('INSERT INTO rows(sample_id,language,official_split,source_row,is_member,membership,content_sha256,content_chars) VALUES(?,?,?,?,?,?,?,?)', batch)
                            con.executemany('INSERT INTO excluded VALUES(?,?,?,?,?,?,?)', bad)
                            con.commit(); batch.clear(); bad.clear()
                            self.monitor.tick(f'index:{language}/{split}', i + 1)
                            self.monitor.guard()
                    con.executemany('INSERT INTO rows(sample_id,language,official_split,source_row,is_member,membership,content_sha256,content_chars) VALUES(?,?,?,?,?,?,?,?)', batch)
                    con.executemany('INSERT INTO excluded VALUES(?,?,?,?,?,?,?)', bad)
                    if set(counts['labels_before']) != {'member', 'non-member'}:
                        raise ValueError(f'Missing class in {language}/{split}')
                    con.execute('INSERT INTO complete VALUES(?,?,?)', (language, split, json.dumps(counts)))
                self.export_quality()
                if counts['reasons'] and self.cfg['invalid_content_policy'] == 'error':
                    raise ValueError(f'Invalid content in {language}/{split}; see input-quality artifacts')
        with self.connect() as con:
            conflict = con.execute('SELECT content_sha256 FROM rows GROUP BY content_sha256 HAVING MIN(is_member)!=MAX(is_member) LIMIT 1').fetchone()
        if conflict:
            raise ValueError(f'Identical valid content has conflicting labels: {conflict[0]}')
        self.export_quality()

    def export_quality(self):
        import pandas as pd
        with self.connect() as con:
            quality = {f'{a}/{b}': json.loads(c) for a, b, c in con.execute('SELECT * FROM complete')}
            # Stream exclusions to CSV; do not materialize all source rows.
            with (self.out / 'excluded_content_rows.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['sample_id', 'language', 'official_split', 'source_row', 'membership', 'reason', 'content_type'])
                writer.writerows(con.execute('SELECT * FROM excluded ORDER BY language,official_split,source_row'))
        write_json(self.out / 'data_quality.json', {'policy': self.cfg['invalid_content_policy'], 'splits': quality,
                                                   'eligible_population': 'nonempty strings; global hash deduplication and full official-test exclusion from development'})
        write_json(self.out / 'dataset_schema.json', {k: v['features'] for k, v in quality.items()})
        if self.cfg['invalid_content_policy'] == 'error' and any(v['reasons'] for v in quality.values()):
            raise ValueError('Strict invalid-content policy rejected source; see exclusions CSV')

    def manifests(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        self.index()
        parts = {}
        columns = ['sample_id', 'language', 'official_split', 'source_row', 'is_member', 'membership', 'content_sha256', 'content_chars']
        with self.connect() as con:
            for split, cap in [('train', self.cfg['dev_per_language']), ('test', self.cfg['test_per_language'])]:
                groups = []
                for language in LANGUAGES:
                    for label in [0, 1]:
                        # Same keep-first global deduplication and stratum sample
                        # ordering as legacy load_data, but metadata-only queries.
                        sql = '''SELECT r.sample_id,r.language,r.official_split,r.source_row,r.is_member,r.membership,r.content_sha256,r.content_chars
                                 FROM rows r WHERE r.official_split=? AND r.language=? AND r.is_member=?
                                 AND r.seq=(SELECT MIN(d.seq) FROM rows d WHERE d.official_split=r.official_split AND d.content_sha256=r.content_sha256)'''
                        if split == 'train':
                            sql += " AND NOT EXISTS(SELECT 1 FROM rows t WHERE t.official_split='test' AND t.content_sha256=r.content_sha256)"
                        sql += ' ORDER BY r.seq'
                        group = pd.DataFrame(con.execute(sql, (split, language, label)).fetchall(), columns=columns)
                        if len(group) < cap // 2:
                            raise ValueError(f'Insufficient eligible {language}/{split}/{label}: {len(group)}')
                        groups.append(group.sample(cap // 2, random_state=self.cfg['seed']))
                parts[split] = pd.concat(groups, ignore_index=True)
        shadow, calibration = train_test_split(parts['train'], test_size=.2, random_state=self.cfg['seed'],
                                                stratify=parts['train'].language + parts['train'].is_member.astype(str))
        frames = [shadow.reset_index(drop=True), calibration.reset_index(drop=True), parts['test']]
        seen_ids, seen_hashes = set(), set()
        for frame, role in zip(frames, ['shadow', 'calibration', 'heldout_test']):
            ids, hashes = set(frame.sample_id), set(frame.content_sha256)
            if len(ids) != len(frame) or len(hashes) != len(frame) or seen_ids & ids or seen_hashes & hashes:
                raise ValueError('Duplicate/overlapping sample identity or content across roles')
            if not frame.official_split.eq('test' if role == 'heldout_test' else 'train').all():
                raise ValueError('Official development/test split violation')
            seen_ids.update(ids); seen_hashes.update(hashes)
            frame['split_role'] = role
        quality_path = self.out / 'data_quality.json'
        quality = json.loads(quality_path.read_text())
        quality.update(status='complete', sampled_rows=dict(zip(
            ['shadow', 'calibration', 'heldout_test'], [len(frame) for frame in frames])))
        write_json(quality_path, quality)
        return tuple(frames)

    def batches(self, manifest, size):
        """Yield selected text only, with original manifest position. Cache capped."""
        import pandas as pd
        manifest = manifest.reset_index(drop=True)
        needed = {row.sample_id: (i, row) for i, row in enumerate(manifest.itertuples(index=False))}
        cap = int(self.cfg['resources']['text_cache_gib'] * 2**30)
        cache = sqlite3.connect(self.text_db)
        cache.execute('PRAGMA cache_size=-8192')
        cache.execute('CREATE TABLE IF NOT EXISTS texts (sample_id TEXT PRIMARY KEY, content_hash TEXT, value BLOB)')
        buffer = []
        def attach(sid, text):
            pos, meta = needed.pop(sid)
            if digest(text) != meta.content_sha256:
                raise ValueError(f'Content changed for stable ID {sid}')
            record = meta._asdict(); record.update(content=text, manifest_position=pos)
            buffer.append(record)
        try:
            for sid, sha, packed in cache.execute('SELECT * FROM texts'):
                if sid in needed:
                    if sha != needed[sid][1].content_sha256:
                        raise ValueError('Text-cache identity mismatch')
                    attach(sid, zlib.decompress(packed).decode('utf-8'))
                    if len(buffer) >= size:
                        yield pd.DataFrame(buffer); buffer = []
            for language in LANGUAGES:
                for split in ['train', 'test']:
                    wanted = {sid for sid in needed if sid.startswith(f'{language}/{split}/')}
                    if not wanted:
                        continue
                    for i, row in enumerate(self.source(language, split)):
                        sid = f'{language}/{split}/{i}'
                        if sid not in wanted:
                            continue
                        text = row['content']
                        attach(sid, text); wanted.remove(sid)
                        if cap and self.text_db.stat().st_size < cap:
                            packed = zlib.compress(text.encode('utf-8'))
                            if self.text_db.stat().st_size + len(packed) + 16384 < cap:
                                self.monitor.guard()
                                cache.execute('INSERT OR REPLACE INTO texts VALUES(?,?,?)', (sid, digest(text), packed)); cache.commit()
                        if len(buffer) >= size:
                            yield pd.DataFrame(buffer); buffer = []
                        if not wanted:
                            break
            if needed:
                raise ValueError(f'Missing source rows: {list(needed)[:5]}')
            if buffer:
                yield pd.DataFrame(buffer)
        finally:
            cache.close()
