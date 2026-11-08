#!/usr/bin/env python3
"""Download the evaluation-only external test sets named in the proposal into data/external/.

  Gan & Ng (2019)      paraphrased SQuAD dev questions   https://github.com/nusnlp/paraphrasing-squad
  Jia & Liang (2017)   Adversarial SQuAD AddSent / AddOneSent (same files the Hugging Face `squad_adversarial`
                       dataset downloads from CodaLab)

All files are SQuAD v1.1 JSON and are scored with the official evaluate-v2.0.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

FILES = {
    "data/external/gan_ng/dev_para.json": "https://raw.githubusercontent.com/nusnlp/paraphrasing-squad/master/datasets/dev_para.json",
    "data/external/gan_ng/dev_orig.json": "https://raw.githubusercontent.com/nusnlp/paraphrasing-squad/master/datasets/dev_orig.json",
    "data/external/gan_ng/adv_para.json": "https://raw.githubusercontent.com/nusnlp/paraphrasing-squad/master/datasets/adv_para.json",
    "data/external/gan_ng/adv_orig.json": "https://raw.githubusercontent.com/nusnlp/paraphrasing-squad/master/datasets/adv_orig.json",
    "data/external/adversarial_squad/AddSent.json": "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/",
    "data/external/adversarial_squad/AddOneSent.json": "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/",
}


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    failed = 0
    for rel, url in FILES.items():
        dst = os.path.join(root, rel)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"exists   {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            print(f"download {rel} <- {url}")
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAILED   {rel}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
