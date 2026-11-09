#!/usr/bin/env python3
"""Evaluate a trained extractor (optionally with the reranker) with the official scorer.

Evaluation sets (--eval_set)
  dev          provided/dev-v1.1.json; paraphrase consistency uses back-translated questions
               (data/paraphrases/dev.jsonl, generated on the fly with MarianMT if missing: ~16 min on CPU, cached)
  gan_dev      Gan & Ng (2019) non-adversarial paraphrased dev questions (1,062); the paired view is the
               original question, so "consistency" = agreement between paraphrased and original questions
  gan_adv      Gan & Ng (2019) adversarial paraphrases (56)
  addsent      Adversarial SQuAD AddSent (3,560)       (Jia & Liang, 2017); consistency via back-translated
  addonesent   Adversarial SQuAD AddOneSent (1,787)    questions as for dev (pass --paraphrase_method none to skip)
  custom       --data_file (and optionally --orig_file for a Gan & Ng-style pairing)

Outputs in --out_dir (default runs/<run>/eval_<set>/): predictions.json, official_eval.json (the scorer's own
output), predictions_rerank.json + official_eval_rerank.json when a reranker is given, candidates.jsonl
(top-k distinct spans), disagreements.jsonl, and results.json (everything, incl. per-question-type metrics).

Examples
  python scripts/evaluate.py --weights weights/bert_kl/best.pt --eval_set dev --reranker weights/bert_kl/reranker.pt
  python scripts/evaluate.py --weights weights/bert_kl/best.pt --eval_set addsent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g17qa.consistency import bootstrap_delta, ensemble_predictions, pair_accuracy  # noqa: E402
from g17qa.data import attach_paraphrases, gold_answers, load_paired_files, load_squad, write_squad_subset  # noqa: E402
from g17qa.decode import QuestionOutput, run_extractor  # noqa: E402
from g17qa.metrics import compute_exact, consistency_metrics, exact_match_score, per_bucket_metrics, question_type, run_official_scorer  # noqa: E402
from g17qa.model import load_checkpoint  # noqa: E402
from g17qa.paraphrase import generate_paraphrase_cache  # noqa: E402
from g17qa.reranker import SELECT_MODES, analyze_changes, apply_reranker, candidate_match, load_reranker, recall_at_ks  # noqa: E402
from g17qa.utils import PROJECT_ROOT, amp_dtype_for, ensure_dir, get_device, write_json  # noqa: E402

EVAL_SETS: Dict[str, Dict[str, Optional[str]]] = {
    "dev": {"file": "provided/dev-v1.1.json", "orig": None, "cache": "data/paraphrases/dev.jsonl"},
    "gan_dev": {"file": "data/external/gan_ng/dev_para.json", "orig": "data/external/gan_ng/dev_orig.json", "cache": None},
    "gan_adv": {"file": "data/external/gan_ng/adv_para.json", "orig": "data/external/gan_ng/adv_orig.json", "cache": None},
    "addsent": {"file": "data/external/adversarial_squad/AddSent.json", "orig": None, "cache": "data/paraphrases/addsent.jsonl"},
    "addonesent": {"file": "data/external/adversarial_squad/AddOneSent.json", "orig": None, "cache": "data/paraphrases/addonesent.jsonl"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--eval_set", choices=list(EVAL_SETS) + ["custom"], default="dev")
    p.add_argument("--data_file", default=None, help="for --eval_set custom")
    p.add_argument("--orig_file", default=None, help="for --eval_set custom: original-question file aligned with --data_file")
    p.add_argument("--paraphrase_cache", default=None, help="JSONL cache for the paired view (defaults per eval set)")
    p.add_argument("--paraphrase_method", choices=["marian", "heuristic", "none"], default="marian",
                   help="how to create missing paraphrases for the consistency metric (cached under data/paraphrases/); "
                        "'none' skips the paired view. Ignored for the Gan & Ng sets, whose pair is the original question")
    p.add_argument("--paraphrase_threshold", type=float, default=0.85)
    p.add_argument("--reranker", default=None, help="weights/<run>/reranker.pt")
    p.add_argument("--rerank_margin", type=float, default=None, help="only rerank candidates within this score of the top-1")
    p.add_argument("--select_mode", choices=list(SELECT_MODES), default="reranker",
                   help="how the final answer is chosen when --reranker is given: reranker = argmax of the reranker (default, original "
                        "behaviour); residual = argmax of log p_ext + alpha * log p_rr; extractor = keep the extractor top-1")
    p.add_argument("--alpha", type=float, default=1.0, help="residual mode: weight of the reranker log-probability (e.g. 0.25 / 0.5 / 1.0)")
    p.add_argument("--override_margin", type=float, default=None,
                   help="confidence gate: only replace the extractor top-1 if the reranker prefers the replacement by more than this many nats")
    p.add_argument("--tag", default="", help="name of this reranker variant; outputs then go to runs/<run>/rerank_variants/<tag>/eval_<set>/ "
                                             "so the default eval_<set> artefacts are never overwritten")
    p.add_argument("--pair_ensemble", action="store_true",
                   help="also score the learning-free paraphrase-ensemble baseline (argmax of log p_q + beta * log p_paraphrase over the "
                        "extractor's own candidate scores; needs the paired view, no reranker needed)")
    p.add_argument("--ensemble_beta", type=float, default=1.0, help="weight of the other view in --pair_ensemble")
    p.add_argument("--n_boot", type=int, default=2000, help="bootstrap resamples for the delta confidence intervals")
    p.add_argument("--n_change_examples", type=int, default=100, help="max FIX and max BREAK examples saved to rerank_changes.jsonl")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--official_script", default=os.path.join(PROJECT_ROOT, "provided", "evaluate-v2.0.py"))
    p.add_argument("--n_disagreements", type=int, default=200)
    return p.parse_args()


def resolve_set(args: argparse.Namespace):
    if args.eval_set == "custom":
        if not args.data_file:
            raise SystemExit("--eval_set custom needs --data_file")
        return args.data_file, args.orig_file, args.paraphrase_cache
    spec = EVAL_SETS[args.eval_set]
    cache = args.paraphrase_cache or spec["cache"]
    if cache and not args.paraphrase_cache and args.paraphrase_method == "heuristic":
        # rule-based pairs get their own default cache so they never mix with the MarianMT cache that real runs resume from
        cache = cache[: -len(".jsonl")] + "_heuristic.jsonl"
    return spec["file"], spec["orig"], cache


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    amp = amp_dtype_for(device, args.precision)
    t0 = time.time()
    data_file, orig_file, cache = resolve_set(args)
    for pth in [data_file] + ([orig_file] if orig_file else []):
        if not os.path.exists(pth):
            raise SystemExit(f"missing {pth}; run download_eval_sets.py (external sets) or check the path")

    model, tokenizer, cfg = load_checkpoint(args.weights, device)
    run_name = cfg.get("run_name") or os.path.basename(os.path.dirname(os.path.abspath(args.weights)))
    default_out = os.path.join("runs", run_name, "rerank_variants", args.tag, f"eval_{args.eval_set}") if args.tag else os.path.join("runs", run_name, f"eval_{args.eval_set}")
    out_dir = ensure_dir(args.out_dir or default_out)
    print(f"extractor {args.weights} ({cfg.get('model')}) | set {args.eval_set} ({data_file}) | device {device} | out {out_dir}")

    # ---- examples and the paired view -------------------------------------------------------
    id_map: Dict[str, str] = {}
    if orig_file:
        examples, id_map = load_paired_files(data_file, orig_file)
        paired = True
        pair_kind = "original question"
    else:
        examples = load_squad(data_file)
        paired = False
        pair_kind = "back-translated paraphrase"
    if args.max_examples:
        examples = examples[: args.max_examples]
    if not orig_file:
        method = args.paraphrase_method
        if cache and method != "none":
            generate_paraphrase_cache(examples, cache, device, method=method, threshold=args.paraphrase_threshold)
        if cache and os.path.exists(cache):
            st = attach_paraphrases(examples, cache)
            paired = st["attached"] > 0
            print(f"paraphrases attached to {st['attached']} / {st['total']} questions ({pair_kind})")
    golds = gold_answers(examples)
    qtypes = {ex.qid: question_type(ex.question) for ex in examples}

    reranker = load_reranker(args.reranker, device) if args.reranker else None
    outputs: Dict[str, QuestionOutput] = run_extractor(
        model, tokenizer, examples, device, batch_size=args.batch_size, max_seq_len=cfg.get("max_seq_len", 384),
        doc_stride=cfg.get("doc_stride", 128), max_query_len=cfg.get("max_query_len", 64), top_k=args.top_k,
        max_answer_len=cfg.get("max_answer_len", 30), with_pairs=paired, want_vecs=reranker is not None, amp_dtype=amp,
    )
    preds = {q: o.pred for q, o in outputs.items()}
    preds_p = {q: o.pred_p for q, o in outputs.items() if o.paired}          # true pairs only (consistency metric)

    def full_paired_view(view_preds: Dict[str, str], base_preds: Dict[str, str]) -> Dict[str, str]:
        """Paired-view predictions for *every* question, for scoring the original-question file of the Gan & Ng
        sets: a question whose 'paraphrase' is identical to the original has no separate pair, so its original-view
        prediction is the base prediction itself."""
        return {q: view_preds.get(q, base_preds[q]) for q in base_preds}

    # ---- official scorer ----------------------------------------------------------------------
    # The scorer needs a prediction for every question in the data file, so it is always given a copy of the data
    # file restricted to the questions that were evaluated (identical to the original when --max_examples is 0).
    scored_main = os.path.join(out_dir, "scored_data.json")
    write_squad_subset(data_file, [e.qid for e in examples], scored_main)
    scored_orig = None
    if orig_file:
        scored_orig = os.path.join(out_dir, "scored_data_orig.json")
        write_squad_subset(orig_file, [id_map[e.qid] for e in examples], scored_orig)

    def score(pred_map: Dict[str, str], tag: str, data_path: str, remap: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        pm = {remap.get(k, k): v for k, v in pred_map.items()} if remap else pred_map
        pred_path = os.path.join(out_dir, f"predictions{tag}.json")
        write_json(pm, pred_path)
        return run_official_scorer(data_path, pred_path, os.path.join(out_dir, f"official_eval{tag}.json"), args.official_script)

    results: Dict[str, Any] = {
        "eval_set": args.eval_set, "data_file": data_file, "orig_file": orig_file, "weights": args.weights, "model": cfg.get("model"),
        "loss_mode": cfg.get("loss_mode"), "reranker_path": args.reranker, "rerank_margin": args.rerank_margin, "top_k": args.top_k,
        "reranker_variant": None if reranker is None else {
            "tag": args.tag, "use_pair_score": reranker.use_pair_score, "scalar_features_used": reranker.feature_names,
            "select_mode": args.select_mode, "alpha": args.alpha, "override_margin": args.override_margin, "rerank_margin": args.rerank_margin,
            "trained_with": getattr(reranker, "meta", {}),
        },
        "n_questions": len(examples), "paired_view": pair_kind if paired else None,
        "n_pairs": sum(1 for o in outputs.values() if o.paired), "official_scorer": args.official_script,
    }
    ext: Dict[str, Any] = {"official": score(preds, "", scored_main)}
    ext["by_question_type"] = per_bucket_metrics(preds, golds, qtypes)
    if paired:
        ext["consistency"] = consistency_metrics(preds, preds_p)
        if orig_file:
            ext["paired_view_official_on_orig_file"] = score(full_paired_view(preds_p, preds), "_orig", scored_orig, id_map)
    # ceiling of any reranker over the extractor's top-k: Recall@k EM and best-F1@k
    all_matches = [candidate_match(o.candidates, golds[q]) for q, o in outputs.items()]
    matches = [m for m, o in zip(all_matches, outputs.values()) if o.candidates]
    if matches:
        ext["candidate_ceiling"] = {
            "top_k": args.top_k, "n": len(matches),
            "recall_em": 100.0 * sum(float(em.any()) for em, _ in matches) / len(examples),
            "best_f1": 100.0 * sum(float(f1.max()) for _, f1 in matches) / len(examples),
        }
        ext["recall_at_k"] = recall_at_ks(
            [{"em": em, "f1": f1} for em, f1 in all_matches],
            [k for k in (1, 3, 5, 10, 20) if k <= args.top_k])
        print(f"ceiling    | Recall@{args.top_k} EM {ext['candidate_ceiling']['recall_em']:.2f} | best-F1@{args.top_k} {ext['candidate_ceiling']['best_f1']:.2f}")
    def paired_extras(vp: Dict[str, str], vpp: Dict[str, str]) -> Dict[str, Any]:
        """Both-views-correct statistics and 95% paired-bootstrap CIs of EM and agreement relative to the extractor."""
        out: Dict[str, Any] = {"delta_ci": {"em": bootstrap_delta([exact_match_score(preds[q], golds[q]) > 0 for q in preds],
                                                                  [exact_match_score(vp[q], golds[q]) > 0 for q in preds], args.n_boot)}}
        if paired:
            ids = [q for q in preds if q in preds_p and q in vpp]
            out["pair_accuracy"] = pair_accuracy(vp, vpp, golds)
            out["delta_ci"]["agreement_em"] = bootstrap_delta([compute_exact(preds[q], preds_p[q]) for q in ids], [compute_exact(vp[q], vpp[q]) for q in ids], args.n_boot)
        return out

    if paired:
        ext["pair_accuracy"] = pair_accuracy(preds, preds_p, golds)
        print(f"pair acc   | extractor: EM(q) {ext['pair_accuracy']['em_q']:.2f} EM(para) {ext['pair_accuracy']['em_paraphrase']:.2f} both correct {ext['pair_accuracy']['both_correct']:.2f}")
    results["extractor"] = ext
    print(f"extractor  | official EM {ext['official']['exact']:.2f} F1 {ext['official']['f1']:.2f} (n={ext['official']['total']})"
          + (f" | consistency EM-agreement {ext['consistency']['agreement_em']:.2f} over {ext['consistency']['pairs']} pairs" if paired else ""))

    if reranker is not None:
        rp, rpp, det = apply_reranker(reranker, outputs, device, args.rerank_margin, mode=args.select_mode, alpha=args.alpha,
                                      override_margin=args.override_margin, return_details=True)
        rr: Dict[str, Any] = {"official": score(rp, "_rerank", scored_main)}
        rr["by_question_type"] = per_bucket_metrics(rp, golds, qtypes)
        if paired:
            rr["consistency"] = consistency_metrics(rp, rpp)
            if orig_file:
                rr["paired_view_official_on_orig_file"] = score(full_paired_view(rpp, rp), "_rerank_orig", scored_orig, id_map)
        # extractor-only vs reranked, and the per-question FIX / BREAK / SAME classification (correct = exact match)
        rr["delta"] = {
            "d_em": rr["official"]["exact"] - ext["official"]["exact"], "d_f1": rr["official"]["f1"] - ext["official"]["f1"],
            "d_agreement_em": (rr["consistency"]["agreement_em"] - ext["consistency"]["agreement_em"]) if paired else None,
        }
        rr.update(paired_extras(rp, rpp))
        rr["changes"], change_rows = analyze_changes(outputs, preds, rp, golds, {e.qid: e.question for e in examples}, det, args.n_change_examples)
        with open(os.path.join(out_dir, "rerank_changes.jsonl"), "w", encoding="utf-8") as f:
            for r in change_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results["reranker"] = rr
        c = rr["changes"]
        print(f"+reranker  | official EM {rr['official']['exact']:.2f} F1 {rr['official']['f1']:.2f}"
              + (f" | consistency EM-agreement {rr['consistency']['agreement_em']:.2f}" if paired else ""))
        print(f"delta      | dEM {rr['delta']['d_em']:+.2f} dF1 {rr['delta']['d_f1']:+.2f}"
              + (f" dAgreement {rr['delta']['d_agreement_em']:+.2f}" if paired else "")
              + f" | FIX {c['FIX']} ({c['FIX_pct']:.2f}%) BREAK {c['BREAK']} ({c['BREAK_pct']:.2f}%) SAME_CORRECT {c['SAME_CORRECT']} SAME_WRONG {c['SAME_WRONG']}"
              + f" | mode {args.select_mode} alpha {args.alpha} gate {args.override_margin} pair_score {reranker.use_pair_score}")

    if args.pair_ensemble:
        if not paired:
            print("--pair_ensemble needs the paired view (paraphrases); skipped")
        else:
            ep, epp = ensemble_predictions(outputs, args.ensemble_beta)
            en: Dict[str, Any] = {"beta": args.ensemble_beta, "official": score(ep, "_pair_ensemble", scored_main)}
            en["consistency"] = consistency_metrics(ep, epp)
            en["delta"] = {"d_em": en["official"]["exact"] - ext["official"]["exact"], "d_f1": en["official"]["f1"] - ext["official"]["f1"],
                           "d_agreement_em": en["consistency"]["agreement_em"] - ext["consistency"]["agreement_em"]}
            en.update(paired_extras(ep, epp))
            en["changes"], _ = analyze_changes(outputs, preds, ep, golds, {}, {}, 0)
            if orig_file:
                en["paired_view_official_on_orig_file"] = score(full_paired_view(epp, ep), "_pair_ensemble_orig", scored_orig, id_map)
            results["pair_ensemble"] = en
            print(f"ensemble   | official EM {en['official']['exact']:.2f} F1 {en['official']['f1']:.2f} | agreement {en['consistency']['agreement_em']:.2f} "
                  f"| dEM {en['delta']['d_em']:+.2f} dAgreement {en['delta']['d_agreement_em']:+.2f} | both correct {en['pair_accuracy']['both_correct']:.2f}")

    # ---- candidate dump and qualitative disagreements -------------------------------------------
    with open(os.path.join(out_dir, "candidates.jsonl"), "w", encoding="utf-8") as f:
        for q, o in outputs.items():
            f.write(json.dumps({"id": q, "n_windows": o.n_windows, "candidates": [c.to_json() for c in o.candidates],
                                "candidates_paraphrase": [c.to_json() for c in o.candidates_p]}, ensure_ascii=False) + "\n")
    if paired:
        ex_by = {e.qid: e for e in examples}
        rows = []
        for q, o in outputs.items():
            if o.paired and o.pred != o.pred_p:
                e = ex_by[q]
                rows.append({"id": q, "question": e.question, "paraphrase": e.paraphrase, "pred": o.pred, "pred_paraphrase": o.pred_p, "gold": e.answers, "type": qtypes[q]})
                if len(rows) >= args.n_disagreements:
                    break
        with open(os.path.join(out_dir, "disagreements.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results["elapsed_sec"] = round(time.time() - t0, 1)
    results["extractor_config"] = {k: cfg.get(k) for k in ("model", "loss_mode", "lambda_kl", "lambda_ot", "mu_hung", "max_seq_len", "doc_stride", "epochs", "lr", "val_frac", "split_seed")}
    write_json(results, os.path.join(out_dir, "results.json"))
    print(f"wrote {os.path.join(out_dir, 'results.json')} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
