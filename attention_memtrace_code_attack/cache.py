"""Original transactional store with explicit combined-schema import validation."""
from pathlib import Path
import sqlite3
from reduced_memtrace_code_attack.cache import Cache as TransactionalCache, fingerprint


class Cache(TransactionalCache):
    def __init__(self, path, identity):
        self.scientific_sha256 = fingerprint(identity)
        super().__init__(path)
        try:
            metadata = dict(scientific_sha256=self.scientific_sha256, schema_sha256=identity['schema']['sha256'])
            old = self.get('metadata','combined_identity')
            if old is None and self.con.execute('SELECT COUNT(*) FROM cache').fetchone()[0]:
                raise ValueError('Existing unlabelled/Reduced cache cannot become a combined cache')
            if old is not None and old != metadata:
                raise ValueError('Feature cache has incompatible scientific identity/schema')
            self.put('metadata','combined_identity',metadata)
        except BaseException:
            self.close()
            raise

    def reuse_from(self, path):
        source = TransactionalCache.__new__(TransactionalCache)
        source.path = Path(path)
        source.con = sqlite3.connect(source.path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            metadata = source.get('metadata','combined_identity')
            if metadata != self.get('metadata','combined_identity'):
                raise ValueError('Cannot import Reduced-only or incompatible combined feature cache')
        finally:
            source.close()
        super().reuse_from(path)
