# HimalayaGPT Quantization Comparison

Separate raw benchmark artifacts are preserved for each quantization.

- Shared generation settings: `{"max_tokens": 256, "seed": 1234, "temperature": 0.0}`

| Model | Quantization | Prompt Strict | Inst Strict | Prompt Loose | Inst Loose | Errors | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| himalaya-bf16 | bf16 | 0.2503 | 0.3035 | 0.3141 | 0.3616 | 0 | 831 |
| himalaya-q8 | q8 | 0.2659 | 0.3229 | 0.2984 | 0.3561 | 0 | 831 |
| himalaya-q4 | q4 | 0.1637 | 0.1910 | 0.1817 | 0.2113 | 0 | 831 |

Each row is computed from that model's separate raw `samples.jsonl` benchmark artifact.
