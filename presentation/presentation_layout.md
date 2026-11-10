# CS4248 G17 Presentation and Script Layout

Foundation only: beats and sources, no script prose or slide copy yet.

- **Based on:** branch `ck` at `5ced232`
- **Format:** 15-minute video, all 5 members present, slides submitted with it. No template is prescribed.
- **Due:** Thursday 12 Nov 2026, 11.59pm
- **Length:** 20 slides, 14:30 total, leaving 30s for handovers
- **Source references:** `L` numbers refer to `report/main.tex`. Other paths are relative to `cs4248-project/`; `rep_*.tex` files are in `report/tables/`.

**Storyline:** QA models change their answer when a question is rephrased → training on paraphrase pairs with a KL term makes them steadier, with accuracy within noise in single-seed runs → a second-stage reranker isn't worth adding.

## Speaker 1: Problem and positioning (3:00)

Presenter: TBD

| # | Slide | Time | Script beat | Visual | Source |
|---|---|---|---|---|---|
| 1 | Title | 0:15 | Group, topic, members | Title card | L20–24 |
| 2 | Answers flip under rephrasing | 1:00 | Open on a real example, state why it matters | Example from error table | L36, L92–93 |
| 3 | Research question and contributions | 0:55 | Does KL add stability beyond augmentation? State the novelty honestly | RQ + 3 contributions | L38–40 |
| 4 | Related work | 0:50 | What exists, where we differ | Positioning table | L36–40 |

## Speaker 2: Method (3:00)

Presenter: TBD

| # | Slide | Time | Script beat | Visual | Source |
|---|---|---|---|---|---|
| 5 | System overview | 0:40 | Training vs test-time path | **New pipeline diagram** | `README.md:8–16` |
| 6 | Extractor and decoding | 0:35 | Sliding windows, span head, top-20 spans | Small schematic | L44, `g17qa/decode.py` |
| 7 | Paraphrase generation | 0:45 | Back-translation, similarity filter, yield | Flow + example | L46, `g17qa/paraphrase.py` |
| 8 | Consistency objective | 1:00 | Walk through the loss; CE vs aug vs KL | Equation + 3-row comparison | L48–55, `g17qa/losses.py` |

## Speaker 3: Protocol and main results (3:00)

Presenter: TBD

| # | Slide | Time | Script beat | Visual | Source |
|---|---|---|---|---|---|
| 9 | Evaluation protocol | 0:50 | Held-out articles, test sets, why agreement alone isn't enough | Data-split diagram | L61, L65 |
| 10 | Main results | 1:00 | Agreement up on all backbones, accuracy within noise | Condensed table | `rep_main.tex`, L69 |
| 11 | Augmentation vs KL | 0:40 | Separated on BERT-base only | Stacked bar (new) | L69 |
| 12 | λ choice and OT ablations | 0:30 | Selection rule; OT/dual don't beat KL | `figures/lambda_sweep.png` | L73 |

## Speaker 4: Robustness and second stage (3:00)

Presenter: TBD

| # | Slide | Time | Script beat | Visual | Source |
|---|---|---|---|---|---|
| 13 | Human paraphrases | 0:50 | Most direct test, not from our own pipeline | CE vs KL comparison | L71 |
| 14 | Distractor robustness | 0:40 | Mixed signs, so observations only | Redrawn robustness plot | L75 |
| 15 | Span reranker design | 0:40 | Why try it (oracle headroom), what it does | MLP schematic | L58, `g17qa/reranker.py` |
| 16 | Why no second stage | 0:50 | Reranker vs ensemble; both add-ons need a paraphrase at test time | Condensed table | `rep_second.tex`, L81 |

## Speaker 5: Analysis and close (2:30)

Presenter: TBD

| # | Slide | Time | Script beat | Visual | Source |
|---|---|---|---|---|---|
| 17 | Error analysis | 0:50 | One success, one meaning-changing paraphrase | Error table | L92–96, L103 |
| 18 | Limitations | 0:45 | Single seed, missing DeBERTa aug runs, filter limits | Bullets | L108 |
| 19 | Conclusion | 0:35 | Findings A–D, proposed system | 3–4 takeaways | L106 |
| 20 | Contributions and external code | 0:20 | Ours vs external, thanks | Two columns | L111 |

## To sort out before filling this in

- [ ] **Assign presenters:** name the five speakers above.
- [ ] **Align the README with the report:** `README.md:25–27` credits KL with the full +4.6/+3.8/+3.3, but the report isolates KL only on BERT-base (+1.4). The README title also still includes the reranker, which the report excludes. The script follows the report.
- [ ] **Redraw `figures/robustness.png`:** it has 11 series and uses nearly the same blue for `bert_aug` and `deb_large_kl`.
- [ ] **Avoid `figures/ablation_dev.png` as is:** its x-axis labels overlap, so slide 10 uses a table instead.
- [ ] **Make new visuals:** pipeline diagram, schematics, data-split diagram and the slide 11 stacked bar.
- [ ] **Confirm the submission branch:** `main` is missing the 31 commits on `ck`, so submit from `ck` or merge it first.

**Left out:** t-SNE and loss curves, which aren't in the report either, and a live demo. The README calls `app.py` legacy, and the weights aren't on this machine.
