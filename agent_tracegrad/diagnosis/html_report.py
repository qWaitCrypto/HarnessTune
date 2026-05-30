"""Static HTML report rendering for diagnosis and landscape results."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

from agent_tracegrad.diagnosis.landscape import LandscapeResult
from agent_tracegrad.diagnosis.report import _rank_margin, _select_margin
from agent_tracegrad.diagnosis.types import DiagnosisResult, MarginContribution
from agent_tracegrad.target.objective import target_objective_to_dict


def diagnosis_to_html(result: DiagnosisResult) -> str:
    node_sum = _select_margin(result.margin_distributions, grain="node", view_name="sum")
    objective = ""
    if result.contrastive_result is not None and result.contrastive_result.objective is not None:
        objective = target_objective_to_dict(result.contrastive_result.objective)["objective_formula"]
    content = [
        _html_head("Agent TraceGrad Diagnosis"),
        "<body>",
        "<main class='shell'>",
        "<section class='hero diagnosis-hero'>",
        "<p class='eyebrow'>Decision Boundary Debugger</p>",
        "<h1>Failure Diagnosis</h1>",
        f"<p class='lede'>Bad target <code>{_e(result.bad_result.target.target_id)}</code> "
        f"on nodes <code>{_e(', '.join(result.bad_result.target.node_ids))}</code>.</p>",
        "<div class='metric-grid'>",
        _metric_card("Mode", str(result.metadata.get("mode", "unknown"))),
        _metric_card("Confidence", result.confidence_level),
        _metric_card("Objective", objective or "bad action"),
        "</div>",
        "</section>",
    ]
    if node_sum is not None:
        ranked = _rank_margin(node_sum.contributions)[:12]
        content.extend(
            [
                "<section class='panel'>",
                "<div class='section-title'><h2>Decision Boundary</h2>"
                f"<span>Total margin {_fmt(node_sum.total_margin)}</span></div>",
                "<div class='bars'>",
        *[_margin_bar(item, max_abs=max((abs(entry.margin) for entry in ranked), default=1.0)) for item in ranked],
                "</div>",
                "</section>",
            ]
        )
    if result.diagnostic_labels:
        content.extend(["<section class='panel'>", "<h2>Diagnostic Labels</h2>", "<div class='label-grid'>"])
        for label in result.diagnostic_labels:
            content.append(
                "<article class='label-card'>"
                f"<span>{_e(label.confidence)}</span>"
                f"<h3>{_e(label.label_name)}</h3>"
                f"<p>{_e(label.summary)}</p>"
                f"<p class='recommendation'>{_e(label.recommendation)}</p>"
                "</article>"
            )
        content.extend(["</div>", "</section>"])
    if result.evidence:
        content.extend(["<section class='panel'>", "<h2>Evidence Windows</h2>"])
        for evidence in result.evidence:
            if not evidence.report.top_windows:
                continue
            content.append(f"<h3>{_e(evidence.objective_name)}</h3>")
            content.append("<div class='evidence-list'>")
            for window in evidence.report.top_windows[:6]:
                content.append(
                    "<article class='evidence'>"
                    f"<strong>{_e(window.node_id)}</strong>"
                    f"<span>{_fmt(window.score)}</span>"
                    f"<p>{_e(window.text)}</p>"
                    "</article>"
                )
            content.append("</div>")
        content.append("</section>")
    content.extend(["</main>", "</body>", "</html>"])
    return "\n".join(content)


def landscape_to_html(result: LandscapeResult) -> str:
    content = [
        _html_head("Agent TraceGrad Landscape"),
        "<body>",
        "<main class='shell'>",
        "<section class='hero landscape-hero'>",
        "<p class='eyebrow'>Failure Gradient Field</p>",
        "<h1>Harness Landscape</h1>",
        "<p class='lede'>Cross-trace attribution over shared harness components only.</p>",
        "<div class='metric-grid'>",
        _metric_card("Traces", str(result.metadata.get("trace_count", len(result.traces)))),
        _metric_card("Top K", str(result.metadata.get("top_k", ""))),
        _metric_card("Clusters", str(len(result.clusters))),
        "</div>",
        "</section>",
        "<section class='panel'>",
        "<div class='section-title'><h2>Harness Components</h2><span>ranked by recurrence and score</span></div>",
        "<div class='landscape-table'>",
        "<div class='table-row table-head'><span>Component</span><span>Kind</span><span>Top-k</span><span>Mean</span></div>",
    ]
    for stat in result.component_stats[:40]:
        content.append(
            "<div class='table-row'>"
            f"<span><code>{_e(stat.component_id)}</code></span>"
            f"<span>{_e(stat.sub_block_kind)}</span>"
            f"<span>{stat.topk_count}</span>"
            f"<span>{_fmt(stat.mean_score)}</span>"
            "</div>"
        )
    content.extend(["</div>", "</section>"])
    if result.clusters:
        content.extend(["<section class='panel'>", "<h2>Failure Mode Groups</h2>", "<div class='cluster-grid'>"])
        for cluster in result.clusters:
            content.append(
                "<article class='cluster-card'>"
                f"<h3>{_e(cluster.cluster_id)}</h3>"
                f"<p>{len(cluster.trace_ids)} traces</p>"
                f"<p>{_e(', '.join(cluster.top_components))}</p>"
                "</article>"
            )
        content.extend(["</div>", "</section>"])
    content.extend(["</main>", "</body>", "</html>"])
    return "\n".join(content)


def write_diagnosis_html(result: DiagnosisResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diagnosis_to_html(result), encoding="utf-8")


def write_landscape_html(result: LandscapeResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(landscape_to_html(result), encoding="utf-8")


def _html_head(title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      --ink: #18201b;
      --muted: #5f6b62;
      --paper: #f5efe2;
      --panel: rgba(255,255,255,.76);
      --bad: #b7412e;
      --good: #20745c;
      --line: rgba(24,32,27,.14);
      --shadow: 0 22px 60px rgba(34, 46, 37, .16);
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
      background:
        radial-gradient(circle at 15% 10%, rgba(183,65,46,.18), transparent 28rem),
        radial-gradient(circle at 90% 0%, rgba(32,116,92,.18), transparent 30rem),
        linear-gradient(135deg, #f8f1df 0%, #e9ddc7 48%, #f4efe6 100%);
    }}
    .shell {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 36px 0 64px; }}
    .hero {{ padding: 44px; border: 1px solid var(--line); border-radius: 34px; box-shadow: var(--shadow); background: rgba(255,255,255,.58); }}
    .hero h1 {{ margin: 0; font-size: clamp(42px, 8vw, 92px); letter-spacing: -.06em; line-height: .88; }}
    .eyebrow {{ margin: 0 0 12px; text-transform: uppercase; letter-spacing: .18em; color: var(--muted); font-size: 12px; }}
    .lede {{ max-width: 760px; font-size: 19px; line-height: 1.55; color: var(--muted); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 14px; margin-top: 26px; }}
    .metric {{ padding: 18px; border-radius: 20px; background: rgba(245,239,226,.82); border: 1px solid var(--line); }}
    .metric span {{ display:block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; }}
    .metric strong {{ display:block; margin-top: 8px; font-size: 20px; overflow-wrap: anywhere; }}
    .panel {{ margin-top: 24px; padding: 26px; border-radius: 28px; background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }}
    .section-title {{ display:flex; justify-content:space-between; gap:20px; align-items:end; }}
    h2 {{ margin: 0 0 18px; font-size: 30px; letter-spacing: -.03em; }}
    h3 {{ margin: 0 0 10px; }}
    code {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .88em; }}
    .bars {{ display:grid; gap:12px; }}
    .bar-row {{ display:grid; grid-template-columns: minmax(180px, 320px) 1fr 90px; gap:14px; align-items:center; }}
    .bar-track {{ height: 16px; background: rgba(24,32,27,.09); border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; }}
    .bad {{ background: var(--bad); }}
    .good {{ background: var(--good); }}
    .label-grid, .cluster-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:14px; }}
    .label-card, .cluster-card, .evidence {{ padding:18px; border-radius:20px; background:#fffaf0; border:1px solid var(--line); }}
    .label-card span {{ color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 11px; }}
    .recommendation {{ color: var(--bad); }}
    .evidence-list {{ display:grid; gap:12px; }}
    .evidence strong, .evidence span {{ display:inline-block; margin-right:12px; }}
    .landscape-table {{ display:grid; gap:8px; }}
    .table-row {{ display:grid; grid-template-columns: minmax(220px,2fr) 1fr .5fr .6fr; gap:12px; padding:12px; border-radius:14px; background:#fffaf0; }}
    .table-head {{ color: var(--muted); text-transform: uppercase; letter-spacing:.1em; font-size:12px; background:transparent; }}
    @media (max-width: 720px) {{
      .hero {{ padding: 28px; }}
      .bar-row, .table-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>"""


def _metric_card(label: str, value: str) -> str:
    return f"<div class='metric'><span>{_e(label)}</span><strong>{_e(value)}</strong></div>"


def _margin_bar(item: MarginContribution, *, max_abs: float) -> str:
    width = min(100.0, abs(item.margin) * 100.0 / max(1e-12, max_abs))
    tone = "bad" if item.margin >= 0.0 else "good"
    return (
        "<div class='bar-row'>"
        f"<code>{_e(item.instance_id)}</code>"
        f"<div class='bar-track'><div class='bar-fill {tone}' style='width:{width:.1f}%'></div></div>"
        f"<strong>{_fmt(item.margin)}</strong>"
        "</div>"
    )


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _e(value: object) -> str:
    return escape(str(value), quote=True)
