
## SSSD
Synthetic Social Score Dataset

#### Description
SSSD generates a HuggingFace-ready JSONL dataset of synthetic “worth” labels for individual profiles using X (Twitter)–only social signals (CPM/CPE–based EMV) plus metadata for human capital, network centrality, and contribution; labels are produced via Gemini function-calling to a strict schema (1–10 sub-scores + confidence) or by a deterministic heuristic when you run in heuristic-only mode.

### Install


### How to
```bash
python infer.py
Spawning 8 workers over 1000 samples → shards: [(0, 125), (125, 250), (250, 375), (375, 500), (500, 625), (625, 750), (750, 875), (875, 1000)]
✓ shard done: worth_dataset.jsonl.part0.jsonl
✓ shard done: worth_dataset.jsonl.part2.jsonl
✓ shard done: worth_dataset.jsonl.part6.jsonl
✓ shard done: worth_dataset.jsonl.part5.jsonl
✓ shard done: worth_dataset.jsonl.part4.jsonl
✓ shard done: worth_dataset.jsonl.part7.jsonl
✓ shard done: worth_dataset.jsonl.part3.jsonl
✓ shard done: worth_dataset.jsonl.part1.jsonl
Merging shards…
✓ Wrote 1000 samples → worth_dataset.jsonl
```

### Default parameters
```text
--out              worth_dataset.jsonl
--n                1000
--model            gemini-2.5-pro        (change to gemini-2.0-flash if you prefer)
--seed             42
--no_llm           False                 (LLM ON by default; requires GOOGLE_GENAI_API_KEY)
--workers          max(1, min(8, (os.cpu_count() or 4)*2))
                   (typically 8 on most machines)
--max_retries      3                     (LLM retries per item)
--retry_backoff    0.5                   (initial seconds; exponential)
--flush_every      50                    (flush every N lines per shard)
--rate_limit_qps   0.0                   (0 = unlimited per worker)

Env var required when --no_llm is NOT set:
GOOGLE_GENAI_API_KEY
```

## configuration of the trainer
```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
```