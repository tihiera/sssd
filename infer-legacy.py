
import os
import json
import random
import argparse
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from tools import submit_scores_tool, SYSTEM_INSTRUCTION
from models import sample_profile
from context import build_user_text
from heuristic_functions import heuristic_scores
from dotenv import load_dotenv

load_dotenv()
API_KEY_ENV = "GOOGLE_GENAI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-pro" # gemini-2.5-pro

def call_llm_scores(client: genai.Client, model: str, user_text: str) -> Optional[Dict[str, Any]]:
    tool = submit_scores_tool()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=types.GenerateContentConfig(
                tools=[tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
                ),
                system_instruction=types.Content(role="system", parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]),
            ),
        )
        # extract function call
        for cand in getattr(resp, "candidates", []) or []:
            parts = getattr(getattr(cand, "content", None), "parts", []) or []
            for p in parts:
                fc = getattr(p, "function_call", None)
                if fc and getattr(fc, "name", "") == "submit_scores":
                    args = dict(fc.args)
                    # coerce
                    for k in ["human","social","network","contribution","confidence"]:
                        if k in args:
                            try: args[k] = int(args[k])
                            except Exception: pass
                    # clip
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
    except Exception as e:
        print(f"[LLM error] {e}")
    return None

def to_hf_record(user_text: str, scores: Dict[str, Any]) -> Dict[str, Any]:
    developer_msg = "Return strict JSON with integer scores for human, social, network, contribution (1-10), a confidence (0-100), and up to 4 short reasons."
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

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic worth-scoring dataset with Gemini + function calling.")
    parser.add_argument("--out", type=str, default="worth_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM and use heuristics only")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    client = None
    if not args.no_llm:
        api_key = os.getenv(API_KEY_ENV)
        if not api_key:
            raise SystemExit(f"Set {API_KEY_ENV} to use Gemini (or pass --no_llm).")
        client = genai.Client(api_key=api_key)

    written = 0
    print(f"Starting generation of {args.n} samples...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            feat = sample_profile(rng)
            user_text = build_user_text(feat)

            scores = None
            if client:
                scores = call_llm_scores(client, args.model, user_text)
            if scores is None:
                # heuristic fallback
                scores = heuristic_scores(feat, rng)

            rec = to_hf_record(user_text, scores)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            
            # Print progress on the same line
            print(f"\rGenerating... {written}/{args.n} ({written/args.n*100:.1f}%)", end="", flush=True)
            
            # Checkpoint every 100 samples
            if written % 100 == 0:
                f.flush()  # Force write to disk
                # Copy current file to checkpoint
                checkpoint_file = f"{args.out}.checkpoint_{written}"
                import shutil
                shutil.copy2(args.out, checkpoint_file)
                print(f"\n📁 Checkpoint saved: {checkpoint_file}")
                print(f"\rGenerating... {written}/{args.n} ({written/args.n*100:.1f}%)", end="", flush=True)

    print(f"\n✓ Wrote {written} samples → {args.out}")

if __name__ == "__main__":
    main()
