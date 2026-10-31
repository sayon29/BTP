"""Transactional, checksummed SQLite checkpoints, one completed branch per commit."""
import hashlib
import json
from pathlib import Path
import sqlite3


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sample_identity(row):
    # Membership is deliberately absent: labels cannot affect prefix or model inputs.
    return {k: row[k] for k in ('sample_id', 'language', 'content_sha256', 'official_split', 'source_row', 'split_role')}


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path)
        self.con.execute('PRAGMA journal_mode=DELETE')
        self.con.execute('PRAGMA synchronous=FULL')
        self.con.execute('CREATE TABLE IF NOT EXISTS cache (kind TEXT, key TEXT, payload TEXT, sha TEXT, PRIMARY KEY(kind,key))')
        if self.con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid SQLite checkpoint')

    def get(self, kind, key):
        row = self.con.execute('SELECT payload,sha FROM cache WHERE kind=? AND key=?', (kind,key)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if fingerprint(payload) != row[1]:
            raise ValueError('Corrupt cache result: '+key)
        return payload

    def put(self, kind, key, payload):
        value = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.con:
            old = self.get(kind, key)
            if old is not None and old != payload:
                raise ValueError('Attempt to overwrite a different valid cached result')
            self.con.execute('INSERT OR IGNORE INTO cache VALUES(?,?,?,?)', (kind,key,value,fingerprint(payload)))

    def reuse_from(self, path):
        source = sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro', uri=True)
        try:
            for kind,key,payload,sha in source.execute('SELECT * FROM cache'):
                value = json.loads(payload)
                if fingerprint(value) != sha:
                    raise ValueError('Corrupt imported cache')
                self.put(kind,key,value)
        finally:
            source.close()

    def close(self):
        self.con.close()
