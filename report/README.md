# ACL 2026 Short Paper: Camera-Ready Manuscript

This directory contains the complete LaTeX source and style files for the CS4248 Group G17 project paper:
**"Optimal Transport and Bipartite Matching for Paraphrase-Consistent Extractive Question Answering"**

The manuscript adheres strictly to the ACL format requirements:
- Maximum 4 pages of main text content (Sections 1 through 7, including Limitations)
- Extra 1 page strictly reserved for references
- Total document length: **strictly $\le 5$ pages**

---

## Directory Contents

| File | Description |
| :--- | :--- |
| `main.tex` | Main LaTeX source code |
| `main.pdf` | Pre-compiled camera-ready publication PDF (verified 5 pages) |
| `custom.bib` | BibTeX bibliography containing 16 peer-reviewed citations |
| `acl.sty` | Official ACL 2026 conference style package |
| `acl_natbib.bst` | Official ACL Natbib bibliography format |
| `figures/` | Vector graphics (`.pdf`) produced by `visualize.py` |
| `table1_ablation_results.tex` | Formatted LaTeX Table 1: Empirical Ablation Benchmark |
| `table2_qualitative_cases.tex` | Formatted LaTeX Table 2: Qualitative Case Studies |

---

## Local Compilation Instructions

### Prerequisites
A working TeX Live installation (2022 or later) with `pdflatex` and `bibtex`.

### Build Commands
Execute from within the `report/` folder:

```bash
# 4-step compilation pass for cross-references, citations, and figure layouts:
pdflatex -interaction=batchmode main.tex
bibtex main
pdflatex -interaction=batchmode main.tex
pdflatex -interaction=batchmode main.tex
```

Alternatively, if you have `latexmk`:
```bash
latexmk -pdf main.tex
```

To clean auxiliary files after building:
```bash
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk missfont.log
```

---

## Overleaf Compilation Instructions

1. Log into [Overleaf](https://www.overleaf.com/) and create a **Blank Project**.
2. Upload the entire contents of this directory (`report/`):
   - `main.tex`
   - `custom.bib`
   - `acl.sty`
   - `acl_natbib.bst`
   - `table1_ablation_results.tex`
   - `table2_qualitative_cases.tex`
   - `figures/` folder with `loss_convergence.pdf`, `ablation_benchmark.pdf`, and `boundary_displacement.pdf`
3. Under Overleaf **Menu** (top-left):
   - **Compiler**: `pdfLaTeX`
   - **TeX Live version**: `2023` or `2024`
   - **Main document**: `main.tex`
4. Click **Recompile**.
