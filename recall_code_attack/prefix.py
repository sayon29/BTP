"""Single fixed policy: seed-42 hash rank, shadow non-members, 128-token prefix."""
import hashlib
import re
from recall_code_attack.cache import fingerprint

PREFIX_TOKENS = 128
SEED = 42


def lexical(text):
    # Ignore whitespace/comments' formatting, retain identifiers and literals.
    return re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]', text)


def shingles(values, width):
    return {tuple(values[i:i+width]) for i in range(len(values)-width+1)}


def substantive_overlap(prefix_ids, target_ids, prefix_text, target_text):
    if shingles(prefix_ids, 32) & shingles(target_ids, 32):
        return True
    p, t = lexical(prefix_text), lexical(target_text)
    if shingles(p, 20) & shingles(t, 20):
        return True
    # Reject complete short candidate inclusion as well.
    width = min(len(p), len(t))
    return bool(width and (shingles(p,width) & shingles(t,width)))


def approved_pool(original):
    from adaptive_token_attack.data import validate_partitions
    validate_partitions(original)
    shadow = original['shadow']
    pool = shadow[shadow.is_member.eq(0)].copy()
    if pool.empty or not pool.official_split.eq('train').all():
        raise ValueError('Require saved shadow non-members only')
    if set(pool.content_sha256) & set(original['heldout'].content_sha256):
        raise ValueError('Prefix pool contains held-out content')
    return pool.sort_values('sample_id').reset_index(drop=True)


def choose_prefix(target, target_text, target_ids, candidates):
    """candidates contain verified metadata, first 128 IDs and their decoded text.

    Target labels are never read. Candidate ranking ignores input order. A fixed
    overlap exclusion is a hygiene constraint, not an alternate selection strategy.
    """
    eligible = []
    for p in candidates:
        if p['official_split'] != 'train' or p['is_member'] != 0 or not p['split_role'].startswith('shadow'):
            raise ValueError('Unapproved prefix source')
        if p['language'] != target['language'] or len(p['prefix_ids']) != PREFIX_TOKENS:
            continue
        if p['sample_id'] == target['sample_id'] or p['content_sha256'] == target['content_sha256']:
            continue
        eligible.append(p)
    ordered = sorted(eligible, key=lambda p: (fingerprint([SEED,target['sample_id'],target['content_sha256'],p['sample_id'],p['content_sha256']]), p['sample_id']))
    seen, rejected = set(), 0
    for p in ordered:
        digest = fingerprint(p['prefix_ids'])
        if digest in seen:
            continue
        seen.add(digest)
        if substantive_overlap(p['prefix_ids'], target_ids, p['prefix_text'], target_text):
            rejected += 1
            continue
        return p, rejected
    raise ValueError('No uncontaminated 128-token same-language shadow prefix for '+target['sample_id'])
