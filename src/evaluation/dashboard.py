"""Static evaluation dashboard and tiny local rating server."""

from __future__ import annotations

import json
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

from src.pipeline_v2.redaction import redact

from .models import EvaluationRunSummary


def generate_dashboard(run_dir: Path, summary: EvaluationRunSummary) -> Path:
    run_dir = run_dir.resolve()
    candidates = []
    labels: dict[str, dict[str, str]] = {}
    for candidate in sorted(summary.candidates, key=lambda item: (item.case_id, item.candidate_id)):
        group = labels.setdefault(candidate.case_id, {})
        group.setdefault(candidate.candidate_id, _letter(len(group)))
        candidates.append({
            **candidate.model_dump(mode="json"),
            "blind_label": f"Candidate {group[candidate.candidate_id]}",
            "code_url": _relative_url(run_dir, candidate.code_path),
            "original_code_url": _relative_url(run_dir, candidate.original_code_path),
            "video_url": _relative_url(run_dir, candidate.video_path),
            "diagnostics_url": _relative_url(run_dir, candidate.diagnostics_path),
            "input_url": _relative_url(run_dir, candidate.input_path),
            "request_urls": [
                value for value in (_relative_url(run_dir, path) for path in candidate.request_paths)
                if value
            ],
            "response_urls": [
                value for value in (_relative_url(run_dir, path) for path in candidate.response_paths)
                if value
            ],
            "frame_urls": [
                value for value in (_relative_url(run_dir, path) for path in candidate.frame_paths[:12])
                if value
            ],
        })
    payload = {
        "evaluation_id": summary.evaluation_id,
        "status": summary.status,
        "track": summary.config.track.value,
        "benchmark": str(summary.config.benchmark_path),
        "total_candidates": summary.total_candidates,
        "completed_candidates": summary.completed_candidates,
        "failed_candidates": summary.failed_candidates,
        "total_cost_usd": summary.total_cost_usd,
        "candidates": candidates,
    }
    data = json.dumps(redact(payload), ensure_ascii=False).replace("</", "<\\/")
    html = _DASHBOARD_HTML.replace("__EVALUATION_DATA__", data)
    path = run_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def serve_dashboard(
    run_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    run_dir = run_dir.resolve()
    if not (run_dir / "dashboard.html").is_file():
        raise ValueError(f"dashboard.html not found in {run_dir}")
    ratings_path = run_dir / "review" / "ratings.jsonl"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlparse(self.path).path != "/api/ratings":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("invalid rating payload size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                record = _validated_rating(payload)
                with lock, ratings_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                body = b'{"saved":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"saved": False, "error": str(exc)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), partial(Handler, directory=str(run_dir)))
    url = f"http://{host}:{port}/dashboard.html"
    print(f"Evaluation dashboard: {url}")
    print("Press Ctrl+C to stop the local review server.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _validated_rating(payload: dict) -> dict:
    candidate_id = str(payload.get("candidate_id", "")).strip()
    case_id = str(payload.get("case_id", "")).strip()
    if not candidate_id or not case_id:
        raise ValueError("candidate_id and case_id are required")
    scores = {}
    for name in ("correctness", "layout", "teaching", "creativity", "timing"):
        value = int(payload.get("scores", {}).get(name, 0))
        if value not in range(1, 6):
            raise ValueError(f"{name} must be between 1 and 5")
        scores[name] = value
    verdict = str(payload.get("verdict", "")).lower()
    if verdict not in {"approve", "reject", "tie"}:
        raise ValueError("verdict must be approve, reject, or tie")
    return redact({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "case_id": case_id,
        "blind_label": str(payload.get("blind_label", ""))[:40],
        "scores": scores,
        "verdict": verdict,
        "notes": str(payload.get("notes", ""))[:2000],
    })


def _relative_url(root: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).resolve()
    try:
        return quote(path.relative_to(root).as_posix())
    except ValueError:
        return None


def _letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index + 1)


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manim Evaluation Review</title>
<style>
:root{--ink:#18211d;--muted:#65716a;--line:#d8ddd9;--paper:#f7f8f6;--white:#fff;--green:#19724b;--red:#b3261e;--amber:#9a6500;--blue:#1769aa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif;letter-spacing:0}
header{background:var(--white);border-bottom:1px solid var(--line);padding:20px 28px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0 0 14px}h3{font-size:16px;margin:0}.sub{color:var(--muted)}
main{max-width:1440px;margin:auto;padding:20px 28px 60px}.summary{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));border:1px solid var(--line);background:var(--white);margin-bottom:18px}.metric{padding:14px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:22px}.metric span{color:var(--muted)}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:14px 0}.tabs button,.toolbar button{border:1px solid var(--line);background:var(--white);padding:8px 12px;cursor:pointer}.tabs button.active{background:var(--ink);color:white;border-color:var(--ink)}select,input,textarea{border:1px solid var(--line);background:white;padding:8px;font:inherit}select{min-width:180px}
.panel{display:none}.panel.active{display:block}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}th{background:#edf1ed;position:sticky;top:0}.ok{color:var(--green);font-weight:650}.bad{color:var(--red);font-weight:650}.warn{color:var(--amber);font-weight:650}a{color:var(--blue)}
.review-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.candidate{background:white;border:1px solid var(--line);border-radius:6px;overflow:hidden}.candidate-head{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}.candidate-body{padding:14px}video{width:100%;aspect-ratio:16/9;background:#000}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:8px}.frames img{width:100%;aspect-ratio:16/9;object-fit:cover;border:1px solid var(--line)}
.findings{max-height:180px;overflow:auto;padding-left:18px}.rating{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin:12px 0}.rating label{color:var(--muted);font-size:12px}.rating select{display:block;min-width:0;width:100%;margin-top:3px}.rating textarea{width:100%;min-height:64px;resize:vertical}.rating-actions{display:flex;gap:7px;margin-top:8px}.rating-actions button{padding:8px 12px;border:1px solid var(--line);background:white;cursor:pointer}.rating-actions .approve{background:var(--green);color:white}.rating-actions .reject{background:var(--red);color:white}.saved{color:var(--green);margin-left:8px}
details{margin-top:10px}pre{white-space:pre-wrap;max-height:260px;overflow:auto;background:#f0f2ef;padding:10px}.empty{padding:28px;background:white;border:1px solid var(--line);color:var(--muted)}
@media(max-width:900px){main,header{padding-left:14px;padding-right:14px}.summary{grid-template-columns:repeat(2,1fr)}.metric{border-bottom:1px solid var(--line)}.review-grid{grid-template-columns:1fr}.rating{grid-template-columns:repeat(2,1fr)}.frames{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header><h1>Manim Evaluation</h1><div class="sub" id="runTitle"></div></header>
<main>
<section class="summary" id="summary"></section>
<div class="tabs toolbar">
  <button class="active" data-tab="overview">Overview</button>
  <button data-tab="review">Blind review</button>
  <button data-tab="artifacts">Artifacts</button>
</div>
<section id="overview" class="panel active">
  <div class="toolbar"><select id="caseFilter"></select><select id="modelFilter"></select></div>
  <div id="overviewTable"></div>
</section>
<section id="review" class="panel">
  <div class="toolbar"><select id="reviewCase"></select></div>
  <div id="reviewGrid" class="review-grid"></div>
</section>
<section id="artifacts" class="panel"><div id="artifactTable"></div></section>
</main>
<script id="evaluationData" type="application/json">__EVALUATION_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('evaluationData').textContent);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ms=v=>v?`${(v/1000).toFixed(1)}s`:'0s';const money=v=>`$${Number(v||0).toFixed(4)}`;const cases=[...new Set(data.candidates.map(x=>x.case_id))];const models=[...new Set(data.candidates.map(x=>x.model.alias))];
document.getElementById('runTitle').textContent=`${data.evaluation_id} | ${data.track} | ${data.status}`;
document.getElementById('summary').innerHTML=[['Completed',data.completed_candidates],['Failed',data.failed_candidates],['Expected',data.total_candidates],['API cost',money(data.total_cost_usd)],['Track',data.track]].map(x=>`<div class="metric"><b>${esc(x[1])}</b><span>${esc(x[0])}</span></div>`).join('');
function options(values,all){return (all?'<option value="">All</option>':'')+values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}
caseFilter.innerHTML=options(cases,true);modelFilter.innerHTML=options(models,true);reviewCase.innerHTML=options(cases,false);
function findings(c){if(!c.findings.length)return '<span class="ok">None</span>';return `<ul class="findings">${c.findings.map(f=>`<li><b>${esc(f.severity.toUpperCase())} ${esc(f.code)}</b>: ${esc(f.message)}</li>`).join('')}</ul>`}
function renderOverview(){const rows=data.candidates.filter(c=>(!caseFilter.value||c.case_id===caseFilter.value)&&(!modelFilter.value||c.model.alias===modelFilter.value));overviewTable.innerHTML=rows.length?`<table><thead><tr><th>Case</th><th>Model</th><th>Status</th><th>Ready</th><th>Errors</th><th>Warnings</th><th>Issue score</th><th>Duration delta</th><th>Cost</th></tr></thead><tbody>${rows.map(c=>`<tr><td>${esc(c.case_id)}</td><td>${esc(c.model.alias)}</td><td class="${c.status==='SUCCEEDED'?'ok':'bad'}">${esc(c.status)}</td><td>${c.metrics.production_ready?'yes':'no'}</td><td>${c.metrics.error_count}</td><td>${c.metrics.warning_count}</td><td>${c.metrics.issue_score}</td><td>${Number(c.metrics.duration_delta_seconds).toFixed(2)}s</td><td>${money(c.metrics.cost_usd)}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No matching candidates.</div>'}
function rating(c){return `<div class="rating">${['correctness','layout','teaching','creativity','timing'].map(n=>`<label>${n}<select data-score="${n}">${[1,2,3,4,5].map(v=>`<option ${v===3?'selected':''}>${v}</option>`).join('')}</select></label>`).join('')}</div><textarea data-notes placeholder="Review notes"></textarea><div class="rating-actions"><button class="approve" data-verdict="approve">Approve</button><button data-verdict="tie">Tie</button><button class="reject" data-verdict="reject">Reject</button><span class="saved"></span></div>`}
function renderReview(){const rows=data.candidates.filter(c=>c.case_id===reviewCase.value);reviewGrid.innerHTML=rows.length?rows.map(c=>`<article class="candidate" data-id="${esc(c.candidate_id)}"><div class="candidate-head"><h3>${esc(c.blind_label)}</h3><span class="${c.metrics.production_ready?'ok':c.status==='FAILED'?'bad':'warn'}">${c.metrics.production_ready?'ready':esc(c.status.toLowerCase())}</span></div><div class="candidate-body">${c.video_url?`<video controls preload="metadata" src="${esc(c.video_url)}"></video>`:'<div class="empty">No rendered video</div>'}<div class="frames">${c.frame_urls.map(u=>`<a href="${esc(u)}" target="_blank"><img loading="lazy" src="${esc(u)}"></a>`).join('')}</div><details><summary>Diagnostics (${c.findings.length})</summary>${findings(c)}</details>${rating(c)}</div></article>`).join(''):'<div class="empty">No candidates for this case.</div>'}
function renderArtifacts(){artifactTable.innerHTML=`<table><thead><tr><th>Case</th><th>Model</th><th>Replicate</th><th>Track</th><th>Artifacts</th><th>Runtime</th></tr></thead><tbody>${data.candidates.map(c=>`<tr><td>${esc(c.case_id)}</td><td>${esc(c.model.provider+'/'+c.model.model)}</td><td>${c.replicate}</td><td>${esc(c.track)}</td><td>${[[c.input_url,'input'],[c.video_url,'video'],[c.code_url,'code'],[c.original_code_url,'original'],[c.diagnostics_url,'diagnostics'],...c.request_urls.map((u,i)=>[u,`request ${i+1}`]),...c.response_urls.map((u,i)=>[u,`response ${i+1}`])].filter(x=>x[0]).map(x=>`<a href="${esc(x[0])}" target="_blank">${x[1]}</a>`).join(' | ')}</td><td>${ms(c.metrics.generation_ms)} generation<br>${ms(c.metrics.render_ms)} render</td></tr>`).join('')}</tbody></table>`}
async function saveRating(button){const card=button.closest('.candidate');const c=data.candidates.find(x=>x.candidate_id===card.dataset.id);const scores={};card.querySelectorAll('[data-score]').forEach(x=>scores[x.dataset.score]=Number(x.value));const payload={candidate_id:c.candidate_id,case_id:c.case_id,blind_label:c.blind_label,scores,verdict:button.dataset.verdict,notes:card.querySelector('[data-notes]').value};const status=card.querySelector('.saved');try{const result=await fetch('/api/ratings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!result.ok)throw new Error('save failed');status.textContent='Saved'}catch(e){status.textContent='Run pdf2manim eval review to save ratings'}}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')});document.addEventListener('click',e=>{if(e.target.dataset.verdict)saveRating(e.target)});caseFilter.onchange=modelFilter.onchange=renderOverview;reviewCase.onchange=renderReview;renderOverview();renderReview();renderArtifacts();
</script>
</body></html>'''
