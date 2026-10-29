#!/usr/bin/env python3
"""Render synthetic leaderboard artifacts in isolated headless Chrome; no training."""
import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study", required=True)
parser.add_argument("--output", required=True, help="New JSON verification record")
args = parser.parse_args()
study, output = Path(args.study).resolve(), Path(args.output).resolve()
if output.exists():
    raise FileExistsError(output)
spec = importlib.util.spec_from_file_location("test_leaderboard", study / "tests/test_leaderboard.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
with tempfile.TemporaryDirectory(prefix="cod-leaderboard-browser-") as temporary:
    folder = Path(temporary)
    for variant, value in [("mean", .1), ("signed", .2), ("single", .3)]:
        for seed in [1, 2, 3]:
            run = fixtures._run(folder / "runs", variant, seed, value=value,
                                test={"S": 1-value, "Fbeta_max": 1-value, "BER_0_5": value})
            fixtures._profile(run)
    report = fixtures.build_leaderboard(fixtures.aggregate(folder / "runs", plan=fixtures._plan(
        folder / "runs", ["mean", "signed", "single"], [1, 2, 3])))
    destination = folder / "report"
    destination.mkdir()
    fixtures.write_leaderboard(destination, report)
    path = destination / "leaderboard.html"
    script = '''<script>
const checks={};
checks.validation_three_variants=document.querySelectorAll('#boards tbody tr').length===3;
checks.validation_order=document.querySelector('#boards tbody tr td:nth-child(2)').textContent==='legacy__mean';
document.getElementById('test').click();
checks.test_metric_default=document.getElementById('metric').value==='S';
checks.test_three_variants=document.querySelectorAll('#boards tbody tr').length===3;
document.getElementById('metric').value='BER_0_5';document.getElementById('metric').onchange();
checks.error_metric_lower_is_better=document.querySelector('#boards tbody tr td:nth-child(2)').textContent==='legacy__mean';
document.getElementById('search').value='signed';document.getElementById('search').oninput();
checks.variant_filter=document.querySelectorAll('#boards tbody tr').length===1;
document.getElementById('search').value='';document.getElementById('search').oninput();
document.getElementById('cost').click();
checks.cost_metric_default=document.getElementById('metric').value==='flops_per_image';
checks.numeric_trailing_zero_preservation=document.querySelector('#boards tbody tr td:nth-child(3)').textContent==='200';
const result=document.createElement('pre');result.id='browser-result';result.textContent=JSON.stringify(checks);document.body.append(result);
</script>'''
    path.write_text(path.read_text() + script)
    command = ['/usr/bin/google-chrome', '--headless', '--disable-gpu', '--no-sandbox',
               '--disable-background-networking', '--disable-extensions',
               f'--user-data-dir={folder / "chrome-profile"}', '--dump-dom', path.as_uri()]
    process = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stderr[-1000:]
    match = re.search(r'<pre id="browser-result">(.*?)</pre>', process.stdout)
    assert match, process.stdout[-1500:]
    checks = json.loads(html.unescape(match.group(1)))
    assert all(checks.values()), checks
    result = {"status": "passed", "checks": checks, "training": "not_run", "inputs": "synthetic artifact fixtures only",
              "browser": subprocess.check_output(['/usr/bin/google-chrome', '--version'], text=True).strip(),
              "browser_profile": "isolated temporary profile, removed after verification",
              "source_sha256": {name: hashlib.sha256((study / name).read_bytes()).hexdigest()
                                for name in ['summarize_study.py', 'fsa_study/leaderboard.py', 'tests/test_leaderboard.py']}}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
