### Table 2: Qualitative Paraphrase Invariance Case Studies

| Context Passage Excerpt | Query Formulations | Baseline Answer | Proposed Dual Answer | Ground Truth |
| :--- | :--- | :--- | :--- | :--- |
| *"...founded on November 26, 1842 by Rev. Edward Sorin, CSC..."* | **q:** When was the university founded?<br>**q':** In what year was the institution established? | **q:** November 26, 1842<br>**q':** 1842 ❌ *(Boundary Shift)* | **q:** 1842<br>**q':** 1842 ✅ *(Consistent)* | 1842 |
| *"...allegedly appeared to Bernadette Soubirous in 1858 in Lourdes..."* | **q:** To whom did the Virgin Mary appear in 1858?<br>**q':** Who allegedly witnessed the 1858 Marian apparitions? | **q:** Bernadette Soubirous<br>**q':** Lourdes ❌ *(Semantic Flip)* | **q:** Bernadette Soubirous<br>**q':** Bernadette Soubirous ✅ *(Consistent)* | Bernadette Soubirous |
| *"...Denver Broncos defeated the Carolina Panthers 24–10..."* | **q:** Which team won Super Bowl 50?<br>**q':** Who were the champions of Super Bowl 50? | **q:** Denver Broncos<br>**q':** Carolina Panthers ❌ *(Antonym Flip)* | **q:** Denver Broncos<br>**q':** Denver Broncos ✅ *(Consistent)* | Denver Broncos |
