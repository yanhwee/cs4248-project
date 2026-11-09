"""G17 extractive QA package: paraphrase-consistent span extractor + learned span reranker.

Modules
-------
data        SQuAD loading, article-level validation split, sliding-window features, paraphrase attachment
paraphrase  MarianMT back-translation with sentence-embedding filtering and on-disk caching
model       Span extractor (AutoModel encoder + start/end head)
losses      CE, symmetric KL, 1-D Wasserstein, Hungarian span-matching consistency losses
decode      Multi-window decoding into top-k distinct candidate spans with reranker features
reranker    Learned span reranker (MLP over the extractor's own representations)
metrics     EM / F1 identical to evaluate-v2.0.py, consistency, question-type buckets, official scorer call
utils       Seeding, device selection, small I/O helpers
"""

__version__ = "0.2.0"
