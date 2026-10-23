"""Inventory every derived-data field and scan this export; never inspect a user's secrets."""
from pathlib import Path
from collections import defaultdict
import argparse
import ipaddress
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_FIELDS = {'sex', 'gender', 'race', 'ethnicity', 'dob', 'date_of_birth', 'birth_date',
                  'full_name', 'real_name', 'email', 'address', 'phone', 'participant_details', 'age'}
RAW_SAMPLE_FIELDS = {'x', 'y', 'lx', 'ly', 'rx', 'ry', 't', 'xT', 'yT', 'n'}
PATTERNS = {
    'private_filesystem_path': re.compile(r'/(?:home/[A-Za-z0-9_.-]+|root|mnt/[a-z])/|[A-Za-z]:[\\/](?:Users|home)[\\/]', re.I),
    'email': re.compile(r'[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'),
    'credential_value': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY'),
    'signed_download_url': re.compile(r'(?i)X-Amz-(?:Signature|Credential|Security-Token)=|[?&]sig=[A-Za-z0-9%]{20,}'),
}
IP = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')
IGNORE = {'.git', '__pycache__', 'review_outputs', '.venv', 'venv', '.mplconfig'}
DENIED_SUFFIXES = {'.pkl', '.pickle', '.bundle', '.csv', '.xlsx', '.jsonl', '.docx', '.zip', '.pyc', '.log', '.status'}


def audit(write=True):
    hits, schemas, npz_inventory = [], [], []
    result_files = sorted((ROOT / 'experiments/results').rglob('*.json'))
    plotting_files = sorted((ROOT / 'paper/figures/data').glob('*.json'))
    for path in result_files + plotting_files:
        obj = json.loads(path.read_text())
        fields = defaultdict(lambda: {'types': set(), 'min_length': None, 'max_length': None})

        def walk(value, field='$'):
            item = fields[field]
            item['types'].add(type(value).__name__)
            if isinstance(value, (dict, list)):
                length = len(value)
                item['min_length'] = length if item['min_length'] is None else min(item['min_length'], length)
                item['max_length'] = length if item['max_length'] is None else max(item['max_length'], length)
            if isinstance(value, dict):
                if len(RAW_SAMPLE_FIELDS.intersection(value)) >= 4 and any(isinstance(v, list) and len(v) > 1000 for v in value.values()):
                    hits.append({'file': path.relative_to(ROOT).as_posix(), 'field': field, 'kind': 'possible_raw_samples'})
                for key, child in value.items():
                    if key.lower() in PRIVATE_FIELDS:
                        hits.append({'file': path.relative_to(ROOT).as_posix(), 'field': field + '.' + key, 'kind': 'private_attribute'})
                    component = '{index}' if re.fullmatch(r'\d+', key) else key
                    walk(child, field + '.' + component)
            elif isinstance(value, list):
                for child in value:
                    walk(child, field + '[]')

        walk(obj)
        schemas.append({'file': path.relative_to(ROOT).as_posix(), 'root_type': type(obj).__name__,
                        'root_keys': list(obj) if isinstance(obj, dict) else None,
                        'fields': [{'path': key, 'types': sorted(value['types']),
                                    **({'min_length': value['min_length'], 'max_length': value['max_length']} if value['min_length'] is not None else {})}
                                   for key, value in sorted(fields.items())]})
    for path in sorted((ROOT / 'experiments/results').rglob('*.npz')):
        arrays = []
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'mid', 'Db', 'names'}:
                hits.append({'file': path.relative_to(ROOT).as_posix(), 'kind': 'unexpected_npz_fields'})
            for name in archive.files:
                value = archive[name]
                if value.dtype.hasobject:
                    hits.append({'file': path.relative_to(ROOT).as_posix(), 'field': name, 'kind': 'object_array'})
                arrays.append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                               'nbytes': value.nbytes, 'object_dtype': bool(value.dtype.hasobject),
                               'role': {'mid': 'Measurement-chain member indices', 'Db': 'Aggregated bootstrap coordinate differences: member x bootstrap x hold coordinate',
                                        'names': 'Hold-coordinate labels'}.get(name, 'UNREVIEWED')})
        npz_inventory.append({'file': path.relative_to(ROOT).as_posix(), 'arrays': arrays})
    text_scanned = 0
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in IGNORE for part in relative.parts):
            continue
        if path.is_symlink():
            hits.append({'file': relative.as_posix(), 'kind': 'symlink'})
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in DENIED_SUFFIXES:
            hits.append({'file': relative.as_posix(), 'kind': 'excluded_file_type'})
        if path.suffix in {'.npz', '.png', '.pdf'}:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        text_scanned += 1
        for kind, pattern in PATTERNS.items():
            # This scanner's regex literals are rules, not leaked values.
            scan_text = '\n'.join(line for line in text.splitlines() if 're.compile(' not in line)
            if pattern.search(scan_text):
                hits.append({'file': relative.as_posix(), 'kind': kind})
        for candidate in IP.findall(text):
            try:
                ipaddress.IPv4Address(candidate)
            except ipaddress.AddressValueError:
                continue
            hits.append({'file': relative.as_posix(), 'kind': 'ipv4_address'})
            break
    inventory = {'schema_version': 1, 'research_result_json_count': len(result_files),
                 'plotting_json_count': len(plotting_files), 'json_schemas': schemas, 'npz_files': npz_inventory}
    report = {'schema_version': 1, 'status': 'PASS' if not hits else 'FAIL',
              'research_result_json_count': len(result_files), 'plotting_json_count': len(plotting_files),
              'npz_count': len(npz_inventory), 'text_files_scanned': text_scanned,
              'official_public_pseudonyms_retained': True,
              'raw_gaze_samples_distributed': False, 'private_demographics_distributed': False,
              'new_identity_mapping_distributed': False, 'irreversible_anonymisation_claimed': False,
              'limits': 'An allowlist, complete JSON/NPZ schema review and pattern scan; public pseudonyms and summary features can remain linkable to upstream data.',
              'hits': hits}
    if write:
        (ROOT / 'provenance/data_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        (ROOT / 'provenance/privacy_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('limits',)}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    arguments = parser.parse_args()
    raise SystemExit(0 if audit(not arguments.check_only)['status'] == 'PASS' else 1)
