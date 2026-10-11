### All 31 questions

| mode | hit@1 | hit@3 | hit@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.41 | 0.59 | 0.74 | 0.527 | 2.6 | 3.3 |
| keyword | 0.56 | 0.70 | 0.85 | 0.651 | 0.4 | 0.9 |
| hybrid | 0.56 | 0.74 | 0.89 | 0.667 | 2.2 | 3.3 |
| hybrid_rerank | 0.63 | 0.74 | 0.93 | 0.720 | 3.9 | 5.6 |

| mode | false refusals | key fact in answer | citation hits gold section | unanswerable correctly refused |
|---|---|---|---|---|
| vector | 0.59 | 0.30 | 0.33 | 1.00 |
| keyword | 0.37 | 0.56 | 0.56 | 1.00 |
| hybrid | 0.44 | 0.41 | 0.44 | 1.00 |
| hybrid_rerank | 0.26 | 0.52 | 0.56 | 1.00 |

### Held-out test split (15 questions)

| mode | hit@1 | hit@3 | hit@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.38 | 0.54 | 0.69 | 0.483 | 1.6 | 1.8 |
| keyword | 0.46 | 0.62 | 0.77 | 0.551 | 0.4 | 0.7 |
| hybrid | 0.54 | 0.69 | 0.77 | 0.622 | 3.6 | 3.7 |
| hybrid_rerank | 0.54 | 0.62 | 0.85 | 0.614 | 3.0 | 4.2 |

| mode | false refusals | key fact in answer | citation hits gold section | unanswerable correctly refused |
|---|---|---|---|---|
| vector | 0.77 | 0.23 | 0.23 | 1.00 |
| keyword | 0.46 | 0.46 | 0.46 | 1.00 |
| hybrid | 0.46 | 0.38 | 0.38 | 1.00 |
| hybrid_rerank | 0.31 | 0.38 | 0.38 | 1.00 |
