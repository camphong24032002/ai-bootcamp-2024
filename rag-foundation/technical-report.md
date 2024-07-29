## Prerequisites

Create a .env file to setup the environment variable

``` bash
GROQ_API_KEY=<YOUR-GROQ-API-KEY>
TOKENIZERS_PARALLELISM=false
```

There are some updates in requirements.txt, you need to install some additional packages.

## Formula

### TF-IDF

$$ tf = freq / (freq + k1 * (1-b + b * dl / avgdl ))$$

$$ idf = \log(1 + (N - DF + 0.5) / (DF + 0.5)) $$
$$ \approx \log(1+(N-DF)/DF) = \log(DF/DF + (N-DF)/DF) = \log(N/DF)$$
with N is corpus size and DF is document count

$$ score = tf * idf $$

### Similarity

$$
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}} $$

### Reciprocal Rank Fusion (RRF)

$$
RRF(d) = \sum_{i=1}^{n} \frac{1}{k + rank_i(d)}
$$


## Compare Sparse and Dense Retriever

### Sparse
| Tokenizer | Time | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| albert | 239s | 0.1663 | 0.1676 |
| bert | 141s | 0.1688 | 0.17 |
| roberta | 382s | 0.1574 | 0.1587 |

### Dense
| Model | Time | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 220s | 0.1849 | 0.1867 |
| roberta | 395s | 0.126 | 0.1272 |

### Hybrid
| Model | Time | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| bert & all-MiniLM-L6-v2 | 374s | 0.1913 | 0.1931 |

LLM Model: Llama3-70b-8192

### LLM Sparse
| Tokenizer | Answer F1 | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| albert | 0.0283 | 0.0233 | 0.0235 |
| bert | 0.0261 | 0.0225 | 0.226 |
| roberta | 0.0264 | 0.0207 | 0.209 |

### LLM Dense
| Model | Answer F1 | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 0.0283 | 0.0233 | 0.0235 |
| stsb-roberta-base | 0.0225 | 0.0177 | 0.0178 |

### LLM Hybrid

| Model | Answer | Evidence F1 | Text Evidence Only F1 |
|---|---|---|---|
| bert & all-MiniLM-L6-v2 | 0.0287 | 0.0246 | 0.0248 |


### Summary
- Due to the reduction in generating evidence in generate answers and evidences step, the correctness slightly increases.
- The dense method takes a longer time to compute than the sparse method, but dense has a greater correctness.
- I have tried other models, and those got bad correctness.
- For answering, I only run on 50 tests due to lack of resource. In this experiment, I can't estimate the time because it is affected by calling api time.
- I have implemented the hybrid method, which includes sparse and dense. To experiment, I have chosen the best performance of each method (bert for sparse and all-miniLm-L6-v2 for dense). After the experiment, I have seen that it slightly increased.
- Due to lack of time, I can't experiment on several pairs of sparse and dense models.


## Additional

I have implemented the hybrid.py file based on main.py. This method gets the list ID and score of the sparse and dense methods, then combines the result using Reciprocal Rank Fusion. I have changed the structure of rag pipeline to fit the outcome. Although it currently works, it still has a redundant problem in creating variables.

## Scripts for hybrid

For generating prediction:
```bash
python -m scripts.hybrid    --output_path results/hybrid.jsonl   --force_index True    --retrieval_only True
```

For evaluation:
```bash
python evaluate.py --predictions results/hybrid.jsonl --gold qasper-test-v0.3.json --retrieval_only
```
