# CS4248 G17 — Presentation Script

**Paraphrase-Consistent Extractive Question Answering on SQuAD v1.1**

- **Format:** 15-minute video, 20 slides, all five members present
- **Speaking time:** 14:30, leaving 30 s for handovers
- **Pace:** about 140 spoken words per minute, counting numbers as they are said ("−0.69" is four words); no slide needs more than about 150
- **Numbers:** every figure below is taken from `report/main.pdf` (Tables 1–3 and §1–5)
- **Storyline:** QA models change their answer when a question is rephrased → training on paraphrase pairs with a KL term makes them steadier, with accuracy within noise in single-seed runs → a second-stage reranker isn't worth adding

| Speaker | Part | Slides | Time |
|---|---|---|---|
| Speaker 1 (TBD) | Problem and positioning | 1–4 | 3:00 |
| Speaker 2 (TBD) | Method | 5–8 | 3:00 |
| Speaker 3 (TBD) | Protocol and main results | 9–12 | 3:00 |
| Speaker 4 (TBD) | Robustness and second stage | 13–16 | 3:00 |
| Speaker 5 (TBD) | Analysis and close | 17–20 | 2:30 |

---

## Speaker 1 — Problem and positioning (3:00)

### Slide 1 — Title (0:15)

Hi, we're Group G17: Chong Xin Wei, Ng Jian Yi, Soh Jun Ying Gavin, Wang Chengke and Yee Yan Hwee. We make an extractive QA model give the same answer when a question is reworded.

### Slide 2 — Answers flip under rephrasing (1:00)

Here's the problem, in a real SQuAD dev example. The question asks who holds the record for the oldest quarterback in *any* Super Bowl game. Our back-translation pipeline changes one word: "any" becomes "a". The meaning is the same.

But a plain DeBERTa-v3-large extractor, which scores 95.5 F1 on SQuAD, answers "John Elway" to the original and "Peyton Manning" to the paraphrase. The gold answer is Peyton Manning, so whether the model was right depended only on the wording.

This isn't a one-off. On dev questions with an accepted paraphrase, that same model gives the same answer to both phrasings only 89.4% of the time, so about one pair in ten gets a different answer. Earlier work found the same sensitivity to paraphrases and to distractor sentences. That matters, because the way a user phrases a question shouldn't decide whether they get the right answer.

### Slide 3 — Research question and contributions (0:55)

Consistency training itself isn't new, so our question is a narrow one. Does explicitly regularising the start and end span distributions of a question and its paraphrase add stability beyond simply training on the paraphrases? And does it cost accuracy?

We make three contributions. First, a symmetric-KL consistency objective on span distributions, trained with filtered back-translated paraphrases, and a controlled comparison of plain training, augmentation only, and augmentation plus KL. We can isolate the KL term on BERT-base only. Second, an evaluation on three backbones of different strength, against human paraphrases and distractor sentences, with every decision made on held-out training articles. Third, negative results: optimal-transport variants that don't beat KL, and a second-stage reranker and paraphrase ensemble that don't improve in-distribution accuracy.

### Slide 4 — Related work (0:50)

Four lines of work frame our project. Gan and Ng trained QA models on paraphrased questions and released human-written paraphrase test sets. Their method is our augmentation-only baseline, and their sets are our most direct test. Xie and colleagues, in UDA, regularise a model's output distribution to stay the same under label-preserving changes such as back-translation. We apply that idea to the start and end distributions of an extractive QA model. Jia and Liang built AddSent and AddOneSent, which insert distractor sentences into the passage, and we use them as robustness tests. Finally, RankQA re-ranks candidate answers with a second model, and that is the style of span reranker we try as a second stage.

---

## Speaker 2 — Method (3:00)

### Slide 5 — System overview (0:40)

Now our method, starting with the whole system. At training time, every SQuAD question is back-translated through German, and a similarity filter keeps the good paraphrases. The extractor is then trained on each question and its paraphrase together, with a consistency loss tying their predictions.

At test time the path is much simpler. The system reads only the question and the passage, the extractor scores candidate spans, and the top span is the answer. No paraphrase is needed. The dashed box is the second stage we also tested; we'll come back to why it isn't in the final system.

### Slide 6 — Extractor and decoding (0:35)

The extractor is a standard span model. Each question is paired with sliding windows of the passage, up to 384 tokens, stepping by 128. A linear head gives a start and an end logit for every token; windows without the answer point to the CLS token. At inference, every span up to 30 tokens is scored as start plus end logit, minus the window's null score. We keep the 20 best distinct spans, and the top one is the answer.

### Slide 7 — Paraphrase generation (0:45)

To get paraphrases we back-translate each training question from English to German and back, using the OPUS-MT MarianMT models. We keep a pair only when two checks pass: the paraphrase must differ from the question after normalisation, and the cosine similarity of their MiniLM sentence embeddings must be at least 0.85.

That gives a paraphrase for 82% of training questions and 81.6% of dev questions. On dev, the paraphrases are used only to measure agreement, never for training. The example shows a typical accepted pair: "most recent" becomes "last", and "area" becomes "region".

### Slide 8 — Consistency objective (1:00)

This is the loss. For a question q and its paraphrase q-prime on the same window, we take the start and end distributions over the passage tokens. The loss is the cross-entropy on the question, plus the cross-entropy on the paraphrase, plus lambda times a symmetric KL divergence between the two views' start distributions and between their end distributions.

This one equation gives our three training conditions. The first term alone is plain cross-entropy, our CE baseline. Both cross-entropies with lambda at zero is augmentation only, Gan and Ng's approach. Adding the KL term, with lambda equal to one, is our full recipe. Comparing the last two separates the effect of the extra data from the effect of the consistency term.

KL ignores token order, so as ablations we tried an optimal-transport distance, alone and with a Hungarian matching of top spans. As the results will show, neither beats KL.

---

## Speaker 3 — Protocol and main results (3:00)

### Slide 9 — Evaluation protocol (0:50)

Before the results, how we evaluated. We wanted no test data to shape our choices, so we hold out 5% of the training articles, 23 articles, and make every decision on them: the best checkpoint, the value of lambda, and the reranker's training. SQuAD dev and the external sets are used only for final numbers, with the official scorer.

We report EM and F1, plus agreement: the share of question–paraphrase pairs that get the same answer. Agreement alone isn't enough, because a model that always returns the same wrong span would agree perfectly. So we read it together with accuracy and with F1 on human paraphrases, and for the proposed model we also report how often both answers are exactly correct.

### Slide 10 — Main results (1:00)

These are the main results, all single-seed runs. Agreement rises on every backbone: compared with plain training, the full recipe adds 4.6 points on BERT-base, 3.8 on DeBERTa-v3-base, and 3.3 on DeBERTa-v3-large, which reaches 92.7%.

Accuracy barely moves: no EM or F1 score changes by more than 0.3 points. For the large model, a bootstrap over dev questions gives its EM drop of 0.28 an interval that includes zero, so it's consistent with no change, though that covers the evaluation sample, not training seeds.

What does move accuracy is the backbone, from 88.5 to 95.5 F1. Our proposed system, the large model with the full recipe, gets 90.0 EM and 95.4 F1, and both answers are exactly right for 86.6% of pairs.

### Slide 11 — Augmentation vs KL (0:40)

How much of that gain comes from the KL term, and how much from simply seeing more paraphrases? We can only answer this on BERT-base, the one backbone with an augmentation-only run. There, the paraphrase data alone gives 3.2 points of agreement, and the KL term adds a further 1.4. That 1.4 comes from single runs, so treat it as indicative, not confirmed. For the two DeBERTa models we have no augmentation-only run, so their 3.8 and 3.3 points are for the full recipe and can't be split.

### Slide 12 — λ choice and OT ablations (0:30)

We fixed the rule for lambda before training: the largest lambda whose held-out F1 is within 0.5 of the best. Higher lambda raises agreement and costs a few tenths of F1, so the rule picks lambda equal to one, reused for DeBERTa. On the right, the optimal-transport and dual variants land within 0.2 of augmentation alone; KL is 1.4 higher. So we kept KL.

---

## Speaker 4 — Robustness and second stage (3:00)

### Slide 13 — Human paraphrases (0:50)

The most direct test of our question is Gan and Ng's human paraphrases, because people wrote them, not our pipeline. On DeBERTa-v3-large, F1 is 94.9 on the original questions and 93.8 on the paraphrases, identical with and without our training. But agreement between the two phrasings rises from 92.3% to 95.0%. So the two phrasings get more alike answers, at no cost in paraphrase F1.

On BERT-base, the F1 lost to rephrasing halves, from 2.8 to 1.4, though mostly because original-question F1 dips; paraphrase F1 rises only half a point. Agreement rises from 89.1% to 91.0%.

### Slide 14 — Distractor robustness (0:40)

Distractor sentences hurt a lot: plain BERT-base falls from 88.5 F1 on dev to 64.9 on AddSent. Our training moves distractor F1 by at most 1.2 points, and the signs disagree: up for BERT-base and DeBERTa-v3-large, down for DeBERTa-v3-base. On BERT-base most of the gain comes from the paraphrase data, not the KL term. Agreement, though, rises on every backbone. With single runs and opposite signs, we report these as observations only.

### Slide 15 — Span reranker design (0:40)

Why try a second stage? Because the top-20 list almost always contains the right answer: an exact match for 98.9% of held-out questions and 99.5% of dev questions. So we trained a span reranker. For each candidate, it sees the extractor's span hidden states plus 14 scalar features: scores, rank, length, and optionally three pair features from the paraphrase. A small MLP scores each candidate, trained with listwise softmax cross-entropy on the held-out articles only.

### Slide 16 — Why no second stage (0:50)

It didn't pay off. The rerankers change dev EM by 0.2 to 0.3 points, and every interval includes zero. Pair features buy 1.4 points of agreement, but an earlier run of the same reranker gave 2.4, so one-point differences here are noise. The only accuracy gains that clear zero are on the distractor sets, which isn't the in-distribution accuracy we were aiming for.

The training-free paraphrase ensemble is the stronger consistency device: plus 6.6 points of agreement, at a cost of 0.4 EM. But both it and the pair-feature reranker need a paraphrase at test time. So the proposed system has no second stage.

---

## Speaker 5 — Analysis and close (2:30)

### Slide 17 — Error analysis (0:50)

Back to our opening example. The plain model answered John Elway to the original and switched to Peyton Manning when "any" became "a". The consistency-trained model answers Peyton Manning to both, which is correct.

The second row shows what still goes wrong. Back-translation dropped the word "male", so "the male singer" became "the singer", which is a different question, and the model reasonably answers Beyoncé instead of Bruno Mars. That pair still passed our 0.85 similarity filter. Of 200 saved dev disagreements, 119 have exactly one view correct; we didn't label their causes. Because the filter lets some meaning changes through, 100% agreement is neither realistic nor necessarily desirable.

### Slide 18 — Limitations (0:45)

There are limits on how far these results go. Every configuration was trained once, so differences under half a point, the 1.4-point KL gain, the distractor differences and the 56-question adversarial set are unconfirmed. Our bootstrap intervals cover evaluation samples only, and there is none for agreement. We have no augmentation-only runs for DeBERTa. Agreement is measured only on questions with an accepted paraphrase, and the filter admits some meaning-changing paraphrases. Finally, the 0.5-point tolerance in the lambda rule is load-bearing, and everything is on English SQuAD v1.1.

### Slide 19 — Conclusion (0:35)

To sum up. A: training on paraphrase pairs with a KL consistency term makes QA models clearly more stable: agreement rises by 3.3 to 4.6 points, and on human paraphrases too. B: SQuAD accuracy stays within 0.3 points. C: the pattern holds across backbones, but only on BERT-base can we credit part of it to the KL term itself. D: a learned reranker isn't worth adding. Our proposed system is DeBERTa-v3-large with augmentation and KL, and no second stage.

### Slide 20 — Contributions and external code (0:20)

We wrote the paraphrase pipeline, the consistency objective and its variants, decoding, the reranker, and all evaluation code. External pieces are the libraries, the pre-trained weights, the official scorer and the test sets. No closed-source model or API was used. Thank you!
