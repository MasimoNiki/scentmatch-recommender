"""Generate and analyze a blind, multi-ranker evaluation for ScentMatch."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scentmatch.data.audit import read_jsonl
from scentmatch.evaluation.ranking_v3 import evaluate_ranking_rows
from scentmatch.retrieval.community_text import DEFAULT_ARTIFACT_DIR

DEFAULT_QUERIES = Path("data/eval/ranking_queries_v3.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/evaluation_v4")
MODES = ("semantic", "community", "perception")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_blind_pool(
    queries: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
    per_system_depth: int = 5,
    max_candidates: int = 12,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Round-robin systems, deduplicate candidates, then hide their provenance."""
    query_index = {str(row["query_id"]): row for row in queries}
    run_index = {
        mode: {str(row["query_id"]): row for row in rows}
        for mode, rows in rankings.items()
    }
    public_queries = []
    secret_queries = []
    for query_id, query in query_index.items():
        rng = random.Random(f"{seed}:{query_id}")
        mode_order = sorted(run_index)
        rng.shuffle(mode_order)
        selected: dict[str, dict[str, Any]] = {}
        provenance: dict[str, dict[str, int]] = defaultdict(dict)
        for depth in range(per_system_depth):
            for mode in mode_order:
                results = run_index[mode].get(query_id, {}).get("results", [])
                if depth >= len(results):
                    continue
                result = results[depth]
                perfume_id = str(result["perfume_id"])
                provenance[perfume_id][mode] = int(result.get("rank", depth + 1))
                selected.setdefault(perfume_id, result)
                if len(selected) >= max_candidates:
                    break
            if len(selected) >= max_candidates:
                break

        for mode in mode_order:
            results = run_index[mode].get(query_id, {}).get("results", [])
            for depth, result in enumerate(results[:per_system_depth], start=1):
                perfume_id = str(result["perfume_id"])
                if perfume_id in selected:
                    provenance[perfume_id][mode] = depth

        candidate_features: dict[str, dict[str, float | None]] = defaultdict(dict)
        for mode in sorted(run_index):
            results = run_index[mode].get(query_id, {}).get("results", [])
            for result in results:
                perfume_id = str(result["perfume_id"])
                if perfume_id not in selected:
                    continue
                if (
                    result.get("perception_evidence") is not None
                    and selected[perfume_id].get("perception_evidence") is None
                ):
                    selected[perfume_id]["perception_evidence"] = result[
                        "perception_evidence"
                    ]
                candidate_features[perfume_id][f"score_{mode}"] = result.get(
                    "score"
                )
                for name, value in result.get("components", {}).items():
                    if value is not None:
                        candidate_features[perfume_id][name] = value

        candidates = []
        for perfume_id, result in selected.items():
            candidates.append(
                {
                    "candidate_id": f"{query_id}:{perfume_id}",
                    "perfume_id": perfume_id,
                    "name": result["name"],
                    "brand": result["brand"],
                    "accords": result.get("accords", []),
                    "notes": result.get("notes", {}),
                    "community_evidence": result.get("community_evidence", {}),
                    "perception_evidence": result.get("perception_evidence"),
                }
            )
        rng.shuffle(candidates)
        public_queries.append(
            {
                "query_id": query_id,
                "query_text": query["query_text"],
                "category": query["category"],
                "split": query["split"],
                "candidates": candidates,
            }
        )
        secret_queries.append(
            {
                "query_id": query_id,
                "systems": {
                    mode: [
                        str(row["perfume_id"])
                        for row in run_index[mode].get(query_id, {}).get("results", [])
                    ]
                    for mode in sorted(run_index)
                },
                "candidate_provenance": provenance,
                "candidate_features": candidate_features,
                "candidate_brands": {
                    perfume_id: str(result.get("brand", ""))
                    for perfume_id, result in selected.items()
                },
            }
        )
    public = {
        "schema_version": "4.0",
        "instructions": {
            "relevance": "0 wrong, 1 weak, 2 good, 3 excellent",
            "constraint_violation": "Any explicit request constraint is broken",
            "evidence_quality": "0 unsupported, 1 partial, 2 well supported",
            "unknown": "Use when you cannot judge the fragrance reliably",
        },
        "queries": public_queries,
    }
    secret = {
        "schema_version": "4.0",
        "seed": seed,
        "per_system_depth": per_system_depth,
        "max_candidates": max_candidates,
        "systems": sorted(rankings),
        "queries": secret_queries,
    }
    return public, secret


def _render_html(pool: dict[str, Any]) -> str:
    payload = json.dumps(pool, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape("ScentMatch — blind evaluation v4")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1050px;margin:auto;padding:24px;background:#f6f4f0;color:#211f1c}}
h1{{margin-bottom:4px}} .muted{{color:#6c665e}} .bar{{height:10px;background:#ddd;border-radius:8px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#5e7355}} .query{{background:white;padding:18px;border-radius:14px;margin:20px 0}}
.card{{border:1px solid #d8d3ca;border-radius:10px;padding:14px;margin:12px 0}} .facts{{font-size:13px;color:#514c46}}
.buttons button{{margin:4px 4px 4px 0;padding:7px 11px;border:1px solid #aaa;border-radius:8px;background:#fff;cursor:pointer}}
.buttons button.active{{background:#344b3d;color:white;border-color:#344b3d}} textarea{{width:100%;min-height:42px}}
.actions{{position:sticky;bottom:0;background:#f6f4f0;padding:12px 0;display:flex;gap:8px}} .pill{{font-size:12px;background:#eee9df;padding:3px 7px;border-radius:20px}}
</style></head><body>
<h1>{title}</h1><p class="muted">Системы и их позиции скрыты. Оценивайте соответствие запросу, а не известность бренда.</p>
<div class="bar"><span id="progress"></span></div><p id="progressText" class="muted"></p><main id="app"></main>
<div class="actions"><button id="download">Скачать judgments.jsonl</button><button id="clear">Очистить оценки</button></div>
<script>
const pool={payload}; const key='scentmatch-v4-judgments'; let state=JSON.parse(localStorage.getItem(key)||'{{}}');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function save(){{localStorage.setItem(key,JSON.stringify(state)); progress();}}
function setv(id,k,v){{state[id]=state[id]||{{}};state[id][k]=v;if(k==='relevance'){{state[id].decision='rated';state[id].evidence_quality??=1;}}save();render();}}
function progress(){{const all=pool.queries.flatMap(q=>q.candidates);const done=all.filter(c=>state[c.candidate_id]?.decision).length;document.querySelector('#progress').style.width=(100*done/all.length)+'%';document.querySelector('#progressText').textContent=`Оценено ${{done}} из ${{all.length}}`;}}
function buttons(id,key,values,labels){{return `<div class="buttons">${{values.map((v,i)=>`<button class="${{state[id]?.[key]===v?'active':''}}" onclick='setv(${{JSON.stringify(id)}},${{JSON.stringify(key)}},${{JSON.stringify(v)}})'>${{esc(labels[i])}}</button>`).join('')}}</div>`}}
function evidence(c){{const p=c.community_evidence?.pros||[],n=c.community_evidence?.cons||[],m=c.perception_evidence?[`◎ ${{c.perception_evidence.text}}`]:[];return [...m,...p.map(x=>'✓ '+x.text),...n.map(x=>'△ '+x.text)].slice(0,4).map(esc).join('<br>')||'Нет текстового community evidence';}}
function render(){{document.querySelector('#app').innerHTML=pool.queries.map(q=>`<section class="query"><h2>${{esc(q.query_text)}} <span class="pill">${{esc(q.category)}}</span></h2>${{q.candidates.map(c=>`<article class="card"><h3>${{esc(c.name)}} — ${{esc(c.brand)}}</h3><div class="facts"><b>Аккорды:</b> ${{esc((c.accords||[]).join(', '))}}<br><b>Мнения:</b><br>${{evidence(c)}}</div><p><b>Соответствие запросу</b></p>${{buttons(c.candidate_id,'relevance',[0,1,2,3],['0 — не подходит','1 — слабо','2 — хорошо','3 — отлично'])}}${{buttons(c.candidate_id,'decision',['unknown','rated'],['Не знаю этот аромат','Оценка уверенная'])}}<label><input type="checkbox" ${{state[c.candidate_id]?.constraint_violation?'checked':''}} onchange='setv(${{JSON.stringify(c.candidate_id)}},"constraint_violation",this.checked)'> нарушено явное ограничение</label><p><b>Качество evidence</b></p>${{buttons(c.candidate_id,'evidence_quality',[0,1,2],['0','1','2'])}}<textarea placeholder="Необязательный комментарий" onchange='setv(${{JSON.stringify(c.candidate_id)}},"comment",this.value)'>${{esc(state[c.candidate_id]?.comment||'')}}</textarea></article>`).join('')}}</section>`).join('');progress();}}
document.querySelector('#download').onclick=()=>{{let rows=[];for(const q of pool.queries)for(const c of q.candidates){{const s=state[c.candidate_id];if(!s?.decision)continue;rows.push({{query_id:q.query_id,perfume_id:c.perfume_id,judge_id:'max_v1',decision:s.decision,relevance:s.decision==='rated'?(s.relevance??null):null,constraint_violation:s.constraint_violation??false,evidence_quality:s.decision==='rated'?(s.evidence_quality??null):null,comment:s.comment||''}})}}const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{{type:'application/jsonl'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='judgments_v4.jsonl';a.click();}};
document.querySelector('#clear').onclick=()=>{{if(confirm('Удалить все локальные оценки?')){{state={{}};save();render();}}}};render();
</script></body></html>"""


def generate(args: argparse.Namespace) -> None:
    from scentmatch.retrieval.engine import HybridSearchEngine

    queries = read_jsonl(args.queries.resolve())
    perception_dir = args.perception_artifact_dir.resolve()
    available_modes = list(MODES)
    if not (perception_dir / "metadata.json").exists():
        available_modes.remove("perception")
        print("Perception artifacts not found; generating a two-system pool.")
    engine = HybridSearchEngine(
        profiles_path=args.profiles,
        artifact_dir=args.artifact_dir,
        community_profiles_path=args.community_profiles,
        perception_artifact_dir=(
            perception_dir if "perception" in available_modes else None
        ),
        device=args.device,
    )
    query_embeddings = engine.encode_queries(
        [str(query["query_text"]) for query in queries], args.batch_size
    )
    rankings: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in available_modes
    }
    for query, embedding in zip(queries, query_embeddings, strict=True):
        for mode in available_modes:
            result = engine.rank(
                str(query["query_text"]),
                mode=mode,
                top_k=args.system_depth,
                query_embedding=embedding,
            )
            result.update(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "split": query["split"],
                }
            )
            rankings[mode].append(result)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, rows in rankings.items():
        _write_jsonl(output_dir / f"rankings_{mode}.jsonl", rows)
    public, secret = build_blind_pool(
        queries,
        rankings,
        per_system_depth=args.pool_depth,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    (output_dir / "blind_pool_v4.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "system_manifest_v4.json").write_text(
        json.dumps(secret, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "judging_v4.html").write_text(
        _render_html(public), encoding="utf-8"
    )
    print(f"Blind evaluation created: {output_dir / 'judging_v4.html'}")
    print(f"Queries: {len(queries)}; systems: {', '.join(available_modes)}")


def analyze(args: argparse.Namespace) -> None:
    queries = read_jsonl(args.queries.resolve())
    judgments = [
        row
        for row in read_jsonl(args.judgments.resolve())
        if row.get("decision") == "rated"
        and row.get("relevance") is not None
        and row.get("evidence_quality") is not None
    ]
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    manifest_index = {row["query_id"]: row for row in manifest["queries"]}
    reports = {}
    ranking_rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        query_id = str(query["query_id"])
        row = manifest_index[query_id]
        for mode, ids in row["systems"].items():
            ranking_rows_by_mode[mode].append(
                {
                    "query_id": query_id,
                    "results": [
                        {
                            "perfume_id": perfume_id,
                            "brand": row.get("candidate_brands", {}).get(
                                perfume_id, ""
                            ),
                        }
                        for perfume_id in ids[: args.k]
                    ],
                }
            )
    for mode, rows in sorted(ranking_rows_by_mode.items()):
        reports[mode] = evaluate_ranking_rows(queries, rows, judgments, args.k)

    judgment_index = {
        (str(row["query_id"]), str(row["perfume_id"])): row for row in judgments
    }
    query_index = {str(row["query_id"]): row for row in queries}
    ltr_rows = []
    for query_id, row in manifest_index.items():
        for perfume_id, ranks in row["candidate_provenance"].items():
            judgment = judgment_index.get((query_id, perfume_id))
            if judgment is None:
                continue
            features = dict(row.get("candidate_features", {}).get(perfume_id, {}))
            for mode in manifest["systems"]:
                features[f"rank_{mode}"] = ranks.get(mode, 100)
                features[f"retrieved_{mode}"] = int(mode in ranks)
            ltr_rows.append(
                {
                    "query_id": query_id,
                    "perfume_id": perfume_id,
                    "category": query_index[query_id]["category"],
                    "split": query_index[query_id]["split"],
                    "relevance": judgment["relevance"],
                    "constraint_violation": judgment["constraint_violation"],
                    "evidence_quality": judgment["evidence_quality"],
                    "features": features,
                }
            )
    report = {
        "schema_version": "4.0",
        "rated_judgments": len(judgments),
        "ltr_rows": len(ltr_rows),
        "systems": reports,
        "ready_for_ltr": len(ltr_rows) >= 300,
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(args.ltr_output.resolve(), ltr_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("generate")
    make.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    make.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/processed/perfume_profiles_5000.jsonl"),
    )
    make.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/e5_finetuned")
    )
    make.add_argument(
        "--community-profiles",
        type=Path,
        default=Path("data/augmentation/community_profiles_v2.jsonl"),
    )
    make.add_argument(
        "--perception-artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR
    )
    make.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    make.add_argument("--device", default="cpu")
    make.add_argument("--batch-size", type=int, default=16)
    make.add_argument("--system-depth", type=int, default=10)
    make.add_argument("--pool-depth", type=int, default=5)
    make.add_argument("--max-candidates", type=int, default=15)
    make.add_argument("--seed", type=int, default=42)
    make.set_defaults(handler=generate)

    check = subparsers.add_parser("analyze")
    check.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    check.add_argument("--judgments", type=Path, required=True)
    check.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "system_manifest_v4.json",
    )
    check.add_argument(
        "--report",
        type=Path,
        default=Path("data/interim/evaluation_v4_report.json"),
    )
    check.add_argument(
        "--ltr-output",
        type=Path,
        default=Path("data/training/ltr_judgments_v1.jsonl"),
    )
    check.add_argument("--k", type=int, default=5)
    check.set_defaults(handler=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
