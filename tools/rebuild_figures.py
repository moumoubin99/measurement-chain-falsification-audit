"""Rebuild derived-data figures/tables only; no raw-data access or experiment runs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ['derive_data.py', 'gen_tables.py', 'gen_supp_tables.py', 'gen_supp_family.py',
           'gen_fig2_calibration.py', 'gen_fig3_diagnostics.py', 'gen_fig4_identification.py',
           'gen_fig5_native.py', 'gen_supp_figs.py']


def equal_json(left, right, path='$'):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), 'keys differ: ' + path
        for key in left:
            equal_json(left[key], right[key], path + '.' + key)
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), 'list differs: ' + path
        for index, (first, second) in enumerate(zip(left, right)):
            equal_json(first, second, path + '[' + str(index) + ']')
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and ((math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)), 'numeric value differs: ' + path
    else:
        assert left == right, 'value differs: ' + path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'review_outputs')
    parser.add_argument('--report', type=Path, help='Optional validation receipt path')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT / 'experiments' == output or ROOT / 'paper' == output:
        raise SystemExit('Choose a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='measurement_chain_review_') as temporary:
        scratch = Path(temporary)
        shutil.copytree(ROOT / 'experiments/kincost', scratch / 'experiments/kincost', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'experiments/results', scratch / 'experiments/results')
        figures = scratch / 'paper/figures'
        (figures / 'data').mkdir(parents=True)
        for script in (ROOT / 'paper/figures').glob('*.py'):
            shutil.copyfile(script, figures / script.name)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(scratch / 'mplconfig'))
        for name in SCRIPTS:
            completed = subprocess.run([sys.executable, str(figures / name)], cwd=scratch, env=env,
                                       capture_output=True, text=True, timeout=120)
            if completed.returncode:
                # Show useful diagnostics without disclosing the host's temporary path.
                diagnostic = (completed.stderr or completed.stdout).replace(str(scratch), '<temporary-review-tree>')
                raise RuntimeError(name + ' failed: ' + diagnostic[-3000:])
        original = json.loads((ROOT / 'paper/figures/data/derived.json').read_text())
        rebuilt = json.loads((figures / 'data/derived.json').read_text())
        equal_json(original, rebuilt)
        tables = sorted(figures.glob('TABLE_*.tex'))
        comparisons = []
        for table in tables:
            reference = ROOT / 'paper/figures' / table.name
            matches = reference.read_bytes() == table.read_bytes()
            comparisons.append({'table': table.name, 'byte_identical_to_retained': matches})
            assert matches, 'generated table differs: ' + table.name
        assert len(tables) == 8, 'expected eight generated tables'
        generated_pngs = sorted(figures.glob('*.png'))
        assert len(generated_pngs) == 6, 'expected six data figures'
        generated = []
        for source in sorted(figures.rglob('*')):
            if source.is_file() and source.suffix in ('.png', '.pdf', '.tex', '.json'):
                relative = source.relative_to(figures)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                generated.append({'path': relative.as_posix(), 'bytes': target.stat().st_size,
                                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        for name in ('fig1_overview.svg', 'fig1_overview.png'):
            source = ROOT / 'paper/figures' / name
            if source.exists():
                shutil.copyfile(source, output / name)
        import numpy, scipy, pandas, matplotlib
        report = {'schema_version': 1, 'status': 'PASS', 'scope': 'Retained derived-data reconstruction only',
                  'experiments_run': False, 'raw_data_accessed': False, 'network_accessed': False,
                  'plotting_data_numerically_identical': True, 'numeric_comparison_tolerance': 1e-12,
                  'generated_data_figures': len(generated_pngs), 'generated_tables': len(tables),
                  'table_comparisons': comparisons, 'conceptual_figure': 'Original SVG/PNG copied; no model/image generation',
                  'versions': {name: module.__version__ for name, module in [('numpy', numpy), ('scipy', scipy), ('pandas', pandas), ('matplotlib', matplotlib)]},
                  'generated_files': generated}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key not in ('generated_files',)}, indent=2))


if __name__ == '__main__':
    main()
