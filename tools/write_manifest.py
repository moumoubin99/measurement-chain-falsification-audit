"""Refresh the review-export inventory after intentional package edits."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PARTS = {'.git', '__pycache__', 'review_outputs', '.venv', 'venv', '.mplconfig'}
EXCLUDE_NAMES = {'MANIFEST.json', 'MANIFEST.sha256'}


def main():
    files = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDE_PARTS for part in relative.parts) or path.name in EXCLUDE_NAMES:
            continue
        if path.is_symlink():
            raise ValueError('Linked artifacts are not allowed in the public export')
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        files.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    output = ROOT / 'MANIFEST.json'
    output.write_text(json.dumps({'schema_version': 1, 'release': 'review-export-20261023',
        'scope': 'Derived scientific results, locked source, figures and review documentation; no raw gaze/chats/environments',
        'file_count': len(files), 'total_bytes': sum(item['bytes'] for item in files), 'files': files}, indent=2) + '\n')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / 'MANIFEST.sha256').write_text(digest + '  MANIFEST.json\n')
    print(json.dumps({'manifest_files': len(files), 'total_bytes': sum(item['bytes'] for item in files), 'manifest_sha256': digest}))


if __name__ == '__main__':
    main()
