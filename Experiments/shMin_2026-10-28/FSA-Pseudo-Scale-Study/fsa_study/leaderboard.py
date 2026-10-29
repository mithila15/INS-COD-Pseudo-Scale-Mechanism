"""Dependency-free reports: validation selection and descriptive test/cost ranks.

Ranks are only defined inside an identical protocol AND observed seed set. Model
generator/utilizer choices (including their options) are the compared variants,
not reasons to isolate otherwise comparable experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


LOWER_IS_BETTER = frozenset({
    "MAE", "BER_0_5", "FPR_0_5", "FNR_0_5", "parameters", "latency_median_ms",
    "cuda_peak_allocated_bytes", "macs", "flops", "estimated_noncontraction_flops",
    "estimated_counted_flops", "estimated_total_flops", "macs_per_image", "flops_per_image", "estimated_total_flops_per_image",
})
HIGHER_IS_BETTER = frozenset({
    "S", "wF", "E_adaptive", "E_mean", "E_max", "Fbeta_adaptive", "Fbeta_mean", "Fbeta_max",
    "IoU_0_5", "Dice_0_5", "F1_0_5", "Precision_0_5", "Recall_0_5", "Specificity_0_5",
    "Accuracy_0_5", "BalancedAccuracy_0_5", "BoundaryIoU_0_5", "images_per_second",
})
LABELS = {
    "flops": "Contraction FLOPs / batch (2 x MACs)",
    "flops_per_image": "Contraction FLOPs / image (2 x MACs)",
    "macs": "MACs / batch", "macs_per_image": "MACs / image",
    "estimated_total_flops": "Estimated total FLOPs / batch",
    "estimated_total_flops_per_image": "Estimated total FLOPs / image",
    "estimated_noncontraction_flops": "Estimated noncontraction FLOPs / batch",
    "estimated_counted_flops": "Estimated counted FLOPs / batch (consult coverage)",
    "latency_median_ms": "Median inference latency (ms)",
    "images_per_second": "Images / second", "cuda_peak_allocated_bytes": "Peak CUDA allocated bytes",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:12]


def _direction(metric):
    return "lower" if metric in LOWER_IS_BETTER else "higher" if metric in HIGHER_IS_BETTER else None


def build_leaderboard(summary, test_metric="S"):
    if test_metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER or test_metric in {
            "parameters", "images_per_second", "latency_median_ms", "cuda_peak_allocated_bytes",
            "macs", "flops", "macs_per_image", "flops_per_image", "estimated_noncontraction_flops", "estimated_counted_flops",
            "estimated_total_flops", "estimated_total_flops_per_image"}:
        raise ValueError(f"Unknown test ranking metric: {test_metric}")
    cohorts = defaultdict(list)
    for row in summary["rows"]:
        protocol = dict(row["protocol"])
        options = {key: protocol.pop(key, {}) for key in ("generator_options", "utilizer_options")}
        option_tag = f" [options {_digest(options)}]" if any(options.values()) else ""
        seeds = sorted(row["seeds"])
        expected = summary.get("expected_seeds", {}).get(row["variant"])
        missing = sorted(set(expected or []) - set(seeds))
        if len(seeds) == 1:
            completion = "provisional_single_seed"
        elif missing:
            completion = "provisional_missing_seeds"
        elif expected is not None:
            completion = "planned_seeds_complete"
        else:
            completion = "seed_target_unspecified"
        cohort = {"protocol": protocol, "seeds": seeds}
        key = (row["dataset"], row["metric"], _canonical(cohort))
        # Protocol is stored once per cohort in the leaderboard. A 158-variant
        # cross study must not embed its full source-hash map in every row.
        values = {key: value for key, value in row.items() if key != "protocol"}
        cohorts[key].append({**values, "variant_label": row["variant"] + option_tag,
                            "variant_options": options, "expected_seeds": expected,
                            "missing_seeds": missing, "completion": completion,
                            "cohort_id": _digest(cohort), "direction": _direction(row["metric"])})
    boards = []
    for (dataset, metric, cohort_key), rows in sorted(cohorts.items()):
        direction = _direction(metric)
        # Competition ranks: equal values share a rank (1, 1, 3). No rounding
        # before comparison; display precision cannot create artificial ties.
        rows.sort(key=lambda row: ((-row["mean"] if direction == "higher" else row["mean"]), row["variant_label"]))
        previous = None
        rank = None
        for position, row in enumerate(rows, 1):
            if direction and (previous is None or row["mean"] != previous):
                rank = position
            row["rank"] = rank
            previous = row["mean"]
        board = {"dataset": dataset, "metric": metric, "metric_label": LABELS.get(metric, metric),
                 "kind": "selection" if dataset == "validation" else "cost" if dataset == "inference" else "descriptive_test",
                 "direction": direction, "cohort_id": rows[0]["cohort_id"],
                 **json.loads(cohort_key), "rows": rows}
        boards.append(board)
    boards.sort(key=lambda board: (0 if board["dataset"] == "validation" else 2 if board["dataset"] == "inference" else 1,
                                   board["dataset"], board["metric"], board["cohort_id"]))
    runs = summary.get("runs", [])
    counts = Counter(run["status"] for run in runs)
    progress = {"total_runs": len(runs), "planned_runs": sum(run["planned"] for run in runs),
                "training_complete": sum(run["training_complete"] for run in runs),
                "evaluation_complete": sum(run["evaluation_complete"] for run in runs),
                "profile_complete": sum(run["profile_complete"] for run in runs),
                "not_started": counts["not_started"], "training_incomplete": counts["training_incomplete"],
                "all_stages_complete": counts["complete"]}
    return {"schema_version": 1, "default_test_metric": test_metric,
            "selection_rule": "Lowest mean validation MAE among variants with identical training protocol and seed set. Test results never select a checkpoint or winner.",
            "ranking_rule": "Separate ranks for every dataset, metric, protocol and observed seed set; exact ties share a competition rank. No composite accuracy score and no mixing hardware for inference costs.",
            "uncertainty": "Descriptive seed mean and sample standard deviation; not a significance test. A single seed is provisional. Without a plan, target seed completeness is unknown.",
            "cost_note": "Contraction FLOPs = 2 x MACs. Estimated total FLOPs additionally approximate supported noncontraction operators; consult each profile's coverage. Different hardware/input/timing protocols have separate ranks.",
            "plan": summary.get("plan"), "evaluation_subdir": summary.get("evaluation_subdir", "test"),
            "profile_name": summary.get("profile_name", "profile.json"),
            "progress": progress, "runs": runs, "boards": boards}


def _format(value):
    return "—" if value is None else f"{value:.8g}"


def _markdown(report):
    lines = ["# Experiment leaderboard", "", report["selection_rule"], "", report["ranking_rule"], "",
             report["uncertainty"], "", report["cost_note"], "",
             "The standalone `leaderboard.html` includes every metric and protocol. This Markdown view shows validation MAE, the selected test metric, and principal inference costs.", "",
             "## Progress", "", " | ".join(f"{key}: {value}" for key, value in report["progress"].items()), ""]
    costs = {"parameters", "latency_median_ms", "images_per_second", "cuda_peak_allocated_bytes",
             "macs_per_image", "flops_per_image", "estimated_total_flops_per_image"}
    visible = [board for board in report["boards"] if board["dataset"] == "validation" or
               (board["kind"] == "descriptive_test" and board["metric"] == report["default_test_metric"]) or
               (board["dataset"] == "inference" and board["metric"] in costs)]
    for board in visible:
        arrow = "↓" if board["direction"] == "lower" else "↑" if board["direction"] == "higher" else "unranked"
        lines += [f"## {board['dataset']} · {board['metric_label']} {arrow}", "",
                  f"Cohort `{board['cohort_id']}` · same observed seeds: {board['seeds']}. Ranks only compare rows in this cohort.", "",
                  "| Rank | Experiment | Mean | Sample SD | n | Seeds | Expected seeds | State |",
                  "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |"]
        for row in board["rows"]:
            label = row["variant_label"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['rank'] if row['rank'] is not None else '—'} | {label} | {_format(row['mean'])} | {_format(row['sample_std'])} | {row['n']} | {row['seeds']} | {row['expected_seeds'] if row['expected_seeds'] is not None else 'unknown'} | {row['completion']} |")
        lines += ["", "<details><summary>Comparison protocol</summary>", "", "```json",
                  json.dumps(board["protocol"], indent=2, sort_keys=True), "```", "", "</details>", ""]
    pending = [run for run in report["runs"] if run["status"] != "complete"]
    if pending:
        lines += ["## Pending runs", "", "| Experiment | Seed | State | Run |", "| --- | ---: | --- | --- |"]
        for run in pending:
            lines.append(f"| {run['variant']} | {run['seed']} | {run['status']} | `{run['run']}` |")
        lines.append("")
    if not report["boards"]:
        lines += ["No completed scores are available yet. No training or evaluation was started by generating this report.", ""]
    return "\n".join(lines)


def _html(report):
    # Embedded JSON cannot terminate the script element, and data is only written
    # using textContent, so variant labels and filesystem paths remain inert.
    payload = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pseudo-scale experiment leaderboard</title>
<style>
:root{color-scheme:light dark;font:15px/1.5 system-ui,sans-serif}body{max-width:1280px;margin:32px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin-top:28px}p{max-width:1000px}select,input,button{font:inherit;padding:6px;margin:4px 12px 4px 4px}button{cursor:pointer}.controls{display:flex;flex-wrap:wrap;align-items:center}.card{border:1px solid #8886;border-radius:10px;padding:16px;margin:18px 0;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #8884}th button{border:0;background:none;padding:0;margin:0;font-weight:bold;text-align:left}td.numeric{font-variant-numeric:tabular-nums;text-align:right}.muted{opacity:.75}.state{font-size:13px}.winner{background:#389a5520}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}.progress{padding:14px;background:#4789d51a;border-radius:8px}a{color:inherit}
</style>
<h1>Pseudo-scale experiment leaderboard</h1>
<p id="selection"></p><p id="ranking" class="muted"></p><p id="uncertainty" class="muted"></p>
<div id="progress" class="progress"></div>
<div class="controls"><button id="validation">Validation selection</button><button id="test">Test results</button><button id="cost">Inference cost</button></div>
<div class="controls"><label>Dataset <select id="dataset"></select></label><label>Metric <select id="metric"></select></label><label>Experiment <input id="search" type="search" placeholder="Filter variant name"></label></div>
<p class="muted">Each card is an independent comparison cohort. Click column headings to sort within a cohort. Rank always follows the stated metric direction. Scores are descriptive, not significance claims.</p>
<div id="boards"></div><p id="costnote" class="muted"></p>
<details id="pending"><summary>Run progress and missing experiments</summary><div id="runlist"></div></details>
<p><a href="leaderboard.md">Markdown</a> · <a href="leaderboard.csv">Leaderboard CSV</a> · <a href="leaderboard.json">Leaderboard JSON</a> · <a href="summary.csv">All aggregate metrics CSV</a></p>
<script id="report" type="application/json">""" + payload + """</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('report').textContent);
const $=id=>document.getElementById(id);
const element=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
for(const [id,key] of [['selection','selection_rule'],['ranking','ranking_rule'],['uncertainty','uncertainty'],['costnote','cost_note']])$(id).textContent=data[key];
$('progress').textContent=Object.entries(data.progress).map(([k,v])=>k.replaceAll('_',' ')+': '+v).join(' · ');
const datasets=[...new Set(data.boards.map(b=>b.dataset))];
function options(node,values){node.replaceChildren();for(const value of values)node.append(element('option',value));}
options($('dataset'),datasets);if(datasets.includes('validation'))$('dataset').value='validation';
let sortKey='rank',ascending=true;
function metrics(preferred){const values=[...new Set(data.boards.filter(b=>b.dataset===$('dataset').value).map(b=>b.metric))];options($('metric'),values);const target=preferred||($('dataset').value==='validation'?'MAE':$('dataset').value==='inference'?'flops_per_image':data.default_test_metric);if(values.includes(target))$('metric').value=target;render();}
const format=value=>value===null?'—':Number(value).toLocaleString('en-US',{maximumSignificantDigits:8,useGrouping:false});
function render(){
 const container=$('boards');container.replaceChildren();const visible=data.boards.filter(b=>b.dataset===$('dataset').value&&b.metric===$('metric').value);let count=0;
 for(const board of visible){const rows=board.rows.filter(row=>row.variant_label.toLowerCase().includes($('search').value.toLowerCase()));if(!rows.length)continue;count++;
 const card=element('section',undefined,'card');card.append(element('h2',board.dataset+' · '+board.metric_label+' '+(board.direction==='lower'?'↓':board.direction==='higher'?'↑':'(unranked)')));
 card.append(element('p','Cohort '+board.cohort_id+' · same observed seeds ['+board.seeds.join(', ')+'] · '+rows.length+' variants','muted'));
 const table=element('table'),head=element('thead'),header=element('tr');
 const columns=[['rank','Rank'],['variant_label','Experiment'],['mean','Mean'],['sample_std','Sample SD'],['n','n'],['seeds','Seeds'],['expected_seeds','Expected seeds'],['completion','State']];
 for(const [key,title] of columns){const th=element('th'),button=element('button',title+(key===sortKey?(ascending?' ▲':' ▼'):''));button.onclick=()=>{ascending=key===sortKey?!ascending:true;sortKey=key;render();};th.append(button);header.append(th);}head.append(header);table.append(head);
 rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];let comparison;if(x===null)comparison=y===null?0:1;else if(y===null)comparison=-1;else if(typeof x==='number'&&typeof y==='number')comparison=x-y;else comparison=String(x).localeCompare(String(y));return (ascending?comparison:-comparison)||a.variant_label.localeCompare(b.variant_label);});
 const body=element('tbody');for(const row of rows){const tr=element('tr',undefined,row.rank===1?'winner':'');for(const [key] of columns){let value=row[key];if(key==='mean'||key==='sample_std')value=format(value);else if(Array.isArray(value))value=value.join(', ');else if(value===null)value=key==='expected_seeds'?'unknown':'—';tr.append(element('td',String(value),['mean','sample_std','rank','n'].includes(key)?'numeric':key==='completion'?'state':''));}body.append(tr);}table.append(body);card.append(table);
 const detail=element('details');detail.append(element('summary','Comparison protocol and artifact locations'));detail.append(element('pre',JSON.stringify({protocol:board.protocol,runs:board.rows.map(row=>({variant:row.variant_label,runs:row.runs}))},null,2)));card.append(detail);container.append(card);
 }if(!count)container.append(element('p','No completed scores match this view. Planned or incomplete runs appear below.'));
}
$('dataset').onchange=()=>metrics();$('metric').onchange=render;$('search').oninput=render;
$('validation').onclick=()=>{if(datasets.includes('validation')){$('dataset').value='validation';metrics('MAE');}};
$('test').onclick=()=>{const target=datasets.find(value=>value!=='validation'&&value!=='inference');if(target){$('dataset').value=target;metrics(data.default_test_metric);}};
$('cost').onclick=()=>{if(datasets.includes('inference')){$('dataset').value='inference';metrics('flops_per_image');}};
const runTable=element('table'),runHead=element('tr');for(const title of ['Experiment','Seed','Status','Run'])runHead.append(element('th',title));runTable.append(runHead);for(const run of data.runs){const tr=element('tr');for(const key of ['variant','seed','status','run'])tr.append(element('td',String(run[key])));runTable.append(tr);}$('runlist').append(runTable);metrics();
</script></html>
"""


def write_leaderboard(output, report):
    """Write into the already-created new report directory; never execute jobs."""
    output = Path(output)
    (output / "leaderboard.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "leaderboard.md").write_text(_markdown(report), encoding="utf-8")
    (output / "leaderboard.html").write_text(_html(report), encoding="utf-8")
    columns = ["dataset", "metric", "direction", "cohort_id", "rank", "variant", "variant_label", "mean", "sample_std",
               "n", "seeds", "expected_seeds", "missing_seeds", "completion", "variant_options", "protocol", "runs"]
    with (output / "leaderboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for board in report["boards"]:
            for row in board["rows"]:
                record = {**row, "protocol": {**board["protocol"], **row["variant_options"]}}
                writer.writerow({key: json.dumps(record[key]) if isinstance(record[key], (dict, list)) else record[key] for key in columns})
