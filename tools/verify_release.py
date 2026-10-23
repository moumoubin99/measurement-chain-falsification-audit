"""Check published bytes and the original scientific code lock; run no analysis."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    expected_manifest_hash = (ROOT / 'MANIFEST.sha256').read_text().split()[0]
    assert sha256(ROOT / 'MANIFEST.json') == expected_manifest_hash, 'manifest checksum mismatch'
    for item in manifest['files']:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts, 'unsafe manifest path'
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), 'missing/linked file: ' + item['path']
        assert path.stat().st_size == item['bytes'], 'size mismatch: ' + item['path']
        assert sha256(path) == item['sha256'], 'hash mismatch: ' + item['path']
    lock = json.loads((ROOT / 'experiments/results/R105/lock.json').read_text())
    per_file = {name: sha256(ROOT / 'experiments' / name) for name in lock['code_files']}
    combined = hashlib.sha256(''.join(f'{name}:{value}\n' for name, value in per_file.items()).encode()).hexdigest()
    assert per_file == lock['code_files'], 'original scientific source file differs'
    assert combined == lock['code_hash'], 'combined R105 code hash differs'
    copied = json.loads((ROOT / 'provenance/source_files.json').read_text())['files']
    for item in copied:
        assert sha256(ROOT / item['public_path']) == item['source_sha256'], 'export changed source: ' + item['public_path']
    return {'status': 'PASS', 'manifest_files': len(manifest['files']),
            'original_byte_identical_files': len(copied), 'locked_files': len(per_file),
            'combined_code_sha256': combined, 'experiments_run': False}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), indent=2))
    except (AssertionError, OSError, ValueError, KeyError) as error:
        print('Verification failed: ' + str(error), file=sys.stderr)
        sys.exit(1)
