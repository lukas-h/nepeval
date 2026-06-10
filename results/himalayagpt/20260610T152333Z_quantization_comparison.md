# HimalayaGPT Quantization Comparison

Separate raw benchmark artifacts are preserved for each quantization.

- Shared generation settings: `{"max_tokens": 256, "seed": 1234, "temperature": 0.0}`

| Model | Quantization | Prompt Strict | Inst Strict | Prompt Loose | Inst Loose | Errors | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| himalaya-bf16 | bf16 | 0.2503 | 0.3035 | 0.2828 | 0.3376 | 0 | 831 |
| himalaya-q8 | q8 | 0.2455 | 0.3081 | 0.2852 | 0.3469 | 0 | 831 |
| himalaya-q4 | q4 | 0.1889 | 0.2315 | 0.2166 | 0.2638 | 0 | 831 |

Each row is computed from that model's separate raw `samples.jsonl` benchmark artifact.
