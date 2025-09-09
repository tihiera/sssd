import os
import json
import time
import random
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import submit_scores_tool, SYSTEM_INSTRUCTION
from models import sample_profile
from context import build_user_text
from heuristic_functions import heuristic_scores

load_dotenv()
API_KEY_ENV = "GOOGLE_GENAI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-pro"  # or gemini-2.0-flash

# ------------------ LLM call ------------------

def call_llm_scores(client, model, user_text):
    tool = submit_scores_tool()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY
                )
            ),
            system_instruction=types.Content(
                role="system",
                parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
            ),
        ),
    )
    # extract function call
    for cand in getattr(resp, "candidates", []) or []:
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc and getattr(fc, "name", "") == "submit_scores":
                args = dict(fc.args)

                # coerce + clip
                for k in ["human", "social", "network", "contribution", "confidence"]:
                    if k in args:
                        try:
                            args[k] = int(args[k])
                        except Exception:
                            pass
                args["human"]        = int(max(1, min(10, args.get("human", 5))))
                args["social"]       = int(max(1, min(10, args.get("social", 5))))
                args["network"]      = int(max(1, min(10, args.get("network", 5))))
                args["contribution"] = int(max(1, min(10, args.get("contribution", 5))))
                args["confidence"]   = int(max(0, min(100, args.get("confidence", 60))))

                # reasons
                rs = args.get("reasons") or []
                if isinstance(rs, list):
                    rs = [str(x) for x in rs][:4]
                else:
                    rs = [str(rs)]
                args["reasons"] = rs
                return args

    # if we reached here, the model didn't function-call correctly
    raise RuntimeError("LLM did not return a valid function call to `submit_scores`")

def to_hf_record(user_text, scores):
    developer_msg = ("Return strict JSON with integer scores for human, social, network, "
                     "contribution (1-10), a confidence (0-100), and up to 4 short reasons.")
    final_obj = {
        "human": scores["human"],
        "social": scores["social"],
        "network": scores["network"],
        "contribution": scores["contribution"],
        "confidence": scores["confidence"],
        "reasons": scores.get("reasons", []),
    }
    messages = [
        {"role": "system", "content": "You are WorthScorer. Output strict JSON only."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": json.dumps(final_obj, ensure_ascii=False)},
    ]
    return {
        "reasoning_language": "English",
        "developer": developer_msg,
        "user": user_text,
        "final": final_obj,
        "messages": messages,
    }

# ------------------ worker logic ------------------

def make_client(api_key):
    if not api_key:
        raise SystemExit(f"Set {API_KEY_ENV} to use Gemini (or pass --no_llm).")
    return genai.Client(api_key=api_key)

def generate_one(index, base_seed, model, api_key, use_llm, max_retries, backoff):
    """
    Generate exactly one JSONL line for item `index`, deterministically seeded.
    """
    rng = random.Random(base_seed + index)
    feat = sample_profile(rng)
    user_text = build_user_text(feat)

    if use_llm:
        client = make_client(api_key)  # must exist; otherwise exits
        delay = backoff
        last_err = None
        for _ in range(max_retries):
            try:
                scores = call_llm_scores(client=client, model=model, user_text=user_text)
                rec = to_hf_record(user_text, scores)
                return json.dumps(rec, ensure_ascii=False)
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= 2.0
        # after retries, hard-fail (no silent heuristic fallback)
        raise RuntimeError(f"LLM scoring failed after retries: {last_err}")

    # heuristic-only path
    scores = heuristic_scores(feat, rng)
    rec = to_hf_record(user_text, scores)
    return json.dumps(rec, ensure_ascii=False)

def shard_ranges(n, workers):
    """
    Split [0, n) into `workers` contiguous ranges [(start, end), ...].
    """
    workers = max(1, min(workers, n))
    size = n // workers
    rem = n % workers
    ranges = []
    start = 0
    for w in range(workers):
        extra = 1 if w < rem else 0
        end = start + size + extra
        ranges.append((start, end))
        start = end
    return ranges

def process_shard(shard_id, start, end, args, api_key):
    """
    Generate a shard [start, end) into <out>.part{shard_id}.jsonl
    """
    out_part = f"{args.out}.part{shard_id}.jsonl"
    use_llm = (not args.no_llm)
    written = 0

    per_worker_qps = args.rate_limit_qps
    next_allowed = time.time()

    with open(out_part, "w", encoding="utf-8") as f:
        for i in range(start, end):
            if per_worker_qps > 0:
                now = time.time()
                min_interval = 1.0 / per_worker_qps
                if now < next_allowed:
                    time.sleep(next_allowed - now)
                next_allowed = max(now, next_allowed) + min_interval

            line = generate_one(
                index=i,
                base_seed=args.seed,
                model=args.model,
                api_key=api_key,
                use_llm=use_llm,
                max_retries=args.max_retries,
                backoff=args.retry_backoff,
            )
            f.write(line + "\n")
            written += 1

            if written % args.flush_every == 0:
                f.flush()

    return out_part

def merge_shards(out_path, parts):
    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in parts:
            with open(p, "r", encoding="utf-8") as pf:
                shutil.copyfileobj(pf, out_f)

# ------------------ main ------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel synthetic worth dataset generator (X-only).")
    parser.add_argument("--out", type=str, default="worth_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM and use heuristics only")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) * 2)),
                        help="Parallel workers (threads) – LLM is network-bound, threads are fine")
    parser.add_argument("--max_retries", type=int, default=3, help="Max LLM retries per item")
    parser.add_argument("--retry_backoff", type=float, default=0.5, help="Initial backoff seconds for retries")
    parser.add_argument("--flush_every", type=int, default=50, help="Flush every N lines per shard")
    parser.add_argument("--rate_limit_qps", type=float, default=0.0,
                        help="Per-worker max QPS (0 = unlimited). Useful to avoid API rate limits.")
    args = parser.parse_args()

    api_key = os.getenv(API_KEY_ENV)
    if not args.no_llm and not api_key:
        raise SystemExit(f"Set {API_KEY_ENV} to use Gemini (or pass --no_llm).")

    ranges = shard_ranges(args.n, args.workers)
    print(f"Spawning {len(ranges)} workers over {args.n} samples → shards: {ranges}")

    part_paths = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
        futures = []
        for sid, (start, end) in enumerate(ranges):
            futures.append(ex.submit(process_shard, sid, start, end, args, api_key))
        for fut in as_completed(futures):
            part = fut.result()
            part_paths.append(part)
            print(f"✓ shard done: {part}")

    part_paths.sort(key=lambda p: int(p.rsplit("part", 1)[1].split(".jsonl")[0]))

    print("Merging shards…")
    merge_shards(args.out, part_paths)

    print(f"✓ Wrote {args.n} samples → {args.out}")

if __name__ == "__main__":
    main()