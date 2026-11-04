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
  python evaluate.py --weights weights/bert_kl/best.pt --eval_set dev --reranker weights/bert_kl/reranker.pt
  python evaluate.py --weights weights/bert_kl/best.pt --eval_set addsent
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional


from g17qa.data import attach_paraphrases, gold_answers, load_paired_files, load_squad, write_squad_subset
from g17qa.decode import QuestionOutput, run_extractor
from g17qa.metrics import consistency_metrics, per_bucket_metrics, question_type, run_official_scorer
from g17qa.model import load_checkpoint
from g17qa.paraphrase import generate_paraphrase_cache
from g17qa.reranker import apply_reranker, candidate_match, load_reranker
from g17qa.utils import PROJECT_ROOT, amp_dtype_for, ensure_dir, get_device, write_json

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
    return spec["file"], spec["orig"], (args.paraphrase_cache or spec["cache"])


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
    out_dir = ensure_dir(args.out_dir or os.path.join("runs", run_name, f"eval_{args.eval_set}"))
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
    matches = [candidate_match(o.candidates, golds[q]) for q, o in outputs.items() if o.candidates]
    if matches:
        ext["candidate_ceiling"] = {
            "top_k": args.top_k, "n": len(matches),
            "recall_em": 100.0 * sum(float(em.any()) for em, _ in matches) / len(examples),
            "best_f1": 100.0 * sum(float(f1.max()) for _, f1 in matches) / len(examples),
        }
        print(f"ceiling    | Recall@{args.top_k} EM {ext['candidate_ceiling']['recall_em']:.2f} | best-F1@{args.top_k} {ext['candidate_ceiling']['best_f1']:.2f}")
    results["extractor"] = ext
    print(f"extractor  | official EM {ext['official']['exact']:.2f} F1 {ext['official']['f1']:.2f} (n={ext['official']['total']})"
          + (f" | consistency EM-agreement {ext['consistency']['agreement_em']:.2f} over {ext['consistency']['pairs']} pairs" if paired else ""))

    if reranker is not None:
        rp, rpp = apply_reranker(reranker, outputs, device, args.rerank_margin)
        rr: Dict[str, Any] = {"official": score(rp, "_rerank", scored_main)}
        rr["by_question_type"] = per_bucket_metrics(rp, golds, qtypes)
        if paired:
            rr["consistency"] = consistency_metrics(rp, rpp)
            if orig_file:
                rr["paired_view_official_on_orig_file"] = score(full_paired_view(rpp, rp), "_rerank_orig", scored_orig, id_map)
        results["reranker"] = rr
        print(f"+reranker  | official EM {rr['official']['exact']:.2f} F1 {rr['official']['f1']:.2f}"
              + (f" | consistency EM-agreement {rr['consistency']['agreement_em']:.2f}" if paired else ""))

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
