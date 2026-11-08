# Report (ACL short-paper format)

`main.tex` is the 4-page report (plus one page of references) in the official ACL style (`acl.sty`,
`acl_natbib.bst`, Times 11pt). `main.pdf` is the compiled version committed alongside it.

## How numbers get into the paper

Nothing in the tables is typed by hand:

```bash
python scripts/make_results_report.py     # runs/ -> docs/RESULTS.md + docs/results_digest.json
python scripts/make_report_tables.py      # digest -> report/tables/rep_main.tex, rep_second.tex (the paper's two tables)
python visualize.py --best bert_kl_lam1.0 # figures/ and report/tables/table_*.tex, report/figures/*.pdf (general artefacts)
```

Numbers quoted in the prose follow `docs/RESULTS.md` (one-decimal rounding; differences are differences of rounded
values). The qualitative table is transcribed from `runs/deb_large_ce/eval_dev/disagreements.jsonl` and
`runs/deb_large_kl/eval_dev/{candidates,disagreements}.jsonl`; the question ids are in a comment above the table.
The representation-geometry numbers come from `figures/tsne_bert_ce.json` and `figures/tsne_bert_kl.json`.

## Build

```bash
cd report
tectonic main.tex                                             # one command (brew install tectonic); runs BibTeX itself
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```
or upload `main.tex`, `custom.bib`, `acl.sty`, `acl_natbib.bst` and `tables/` to Overleaf (pdfLaTeX).
`\usepackage[T1]{fontenc}` is required so that XeTeX-based engines (Tectonic) pick up the Times fonts.

## Checklist before submission

- 4 pages of content, references on at most 1 extra page, Times 11pt (do not change the preamble or font sizes).
- All five names in the author block; e-mail addresses may be added to it.
- The "Contributions and external code" section is required by the course and must stay.
- After any change to `runs/`, rerun the three commands above and recompile; cross-check quoted numbers against
  `docs/RESULTS.md`.
