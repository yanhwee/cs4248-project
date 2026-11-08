# Report (ACL short-paper format)

`main.tex` is the 4-page report (plus up to 1 page of references) in the official ACL style (`acl.sty`,
`acl_natbib.bst`, Times 11pt).

## How numbers get into the paper

Nothing is typed by hand. After the experiments:

```bash
python visualize.py --best bert_kl_lam1.0  # reads runs/, writes report/tables/*.tex and report/figures/*.pdf; --best = run for the question-type table
python tsne_pairs.py --weights weights/bert_ce/best.pt --paraphrase_cache data/paraphrases/dev.jsonl
python tsne_pairs.py --weights weights/bert_kl/best.pt --paraphrase_cache data/paraphrases/dev.jsonl
cp figures/tsne_bert_*.pdf report/figures/
```

`main.tex` includes `tables/table_ablation.tex`, `tables/table_robustness.tex`, `tables/table_question_types.tex`
and `figures/*.pdf` when they exist, and prints a red `[TBD: ...]` marker otherwise. Every remaining `\TBD{...}`
in `main.tex` (discussion sentences, e-mail addresses, hardware) must be replaced by hand before submission, and the
`\TBD` macro definition removed.

## Build

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
or upload `main.tex`, `custom.bib`, `acl.sty`, `acl_natbib.bst`, `tables/` and `figures/` to Overleaf (pdfLaTeX).

## Checklist before submission

- 4 pages of content, references on at most 1 extra page, Times 11pt (do not change the preamble).
- All five names in the author block.
- The "Contributions and external code" section is required by the course and must stay.
- Cross-check every number against `runs/<run>/eval_<set>/official_eval*.json` (the scorer's own output).
