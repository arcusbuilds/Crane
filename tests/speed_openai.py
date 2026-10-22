#!/usr/bin/env python3
"""Sequential OpenAI chat speed benchmark; Python standard library only.

Run: python3 tests/speed_openai.py --runs 3
Reports and raw responses default to tests/openai_speed_report.{md,json}.
Token rates use API usage, never an estimate from words or stream chunks.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


PROMPTS = {
    "counting": "Count from 1 to 1000. Output only the numbers, one per line, no other text.",
    "algorithms": "Explain binary search to a beginner in about 600 words, including a worked example and its time complexity.",
    "python": "Write a Python function to merge overlapping intervals, then explain each part and give five example inputs and outputs.",
    "science": "Explain photosynthesis in about 600 words, including the roles of sunlight, water, carbon dioxide, chlorophyll, and glucose.",
    "story": "Write a 600-word story about a small robot that gets lost in a library and finds its way home.",
    "json": "Output a JSON array of ten fictional books. Each object must have title, author, genre, and a description of at least 30 words.",
    "translation": "Translate the following passage into French and then Spanish, preserving all details: Every morning, Maya walked through the old city to reach the train station. The streets were quiet, except for the baker arranging fresh bread by the window. One rainy Tuesday, she noticed a handwritten note beside a blue umbrella. It asked whoever found it to deliver a small package to the library before sunset. Maya checked her watch, took a deep breath, and decided that work could wait for one unusual morning.",
    "arithmetic": "For each integer from 1 to 100, write an equation showing its square. Output one equation per line.",
    "comparison": "Compare arrays and linked lists in about 600 words. Discuss indexing, insertion, deletion, memory layout, and practical examples.",
    "planning": "Create a seven-day study plan for learning basic programming. Give three tasks each day and explain each task in at least 30 words.",
}


def request(opener, url, key, payload=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if payload is None else json.dumps(payload).encode()
    return opener.open(urllib.request.Request(url, data=data, headers=headers), timeout=timeout)


def measure(opener, args, prompt):
    payload = dict(model=args.model, messages=[{"role": "user", "content": prompt}],
                   temperature=args.temperature, max_tokens=args.max_tokens,
                   stream=True, stream_options={"include_usage": True})
    if args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    start = time.perf_counter()
    first = last = None
    usage = {}
    content, reasoning = [], []
    finish = None
    done = False
    with request(opener, args.endpoint, args.api_key, payload, args.timeout) as response:
        # SSE messages end at a blank line; data can span multiple lines.
        event = []
        for line in response:
            line = line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                event.append(line[5:].lstrip())
            elif not line and event:
                data = "\n".join(event)
                event = []
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if text or thought:
                        last = time.perf_counter()
                        if first is None:
                            first = last
                    content.append(text)
                    reasoning.append(thought)
                    finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    tokens = usage.get("completion_tokens")
    # First token is excluded from the measured inter-token generation interval.
    decode = ((tokens - 1) / (last - first)
              if isinstance(tokens, int) and tokens > 1 and first is not None and last > first else None)
    return dict(elapsed_s=elapsed, ttft_s=first - start if first else None,
                completion_tokens=tokens, prompt_tokens=usage.get("prompt_tokens"),
                end_to_end_tps=tokens / elapsed if isinstance(tokens, int) else None,
                decode_tps=decode, finish_reason=finish, stream_done=done,
                status="ok" if done and finish and finish != "length" else "incomplete",
                content="".join(content), reasoning_content="".join(reasoning), usage=usage)


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def number(value):
    return f"{value:.2f}" if value is not None else "N/A"


def save(args, rows):
    metadata = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    endpoint=args.endpoint, model=args.model, runs=args.runs,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    thinking="disabled" if args.no_thinking else "server default")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(dict(metadata=metadata, prompts=PROMPTS, results=rows), indent=2) + "\n")
    lines = ["# Local OpenAI speed benchmark", "", *[f"- {k}: {v}" for k, v in metadata.items()], "",
             "Sequential streaming requests; no warmup. TTFT includes prefill and transport. "
             "End-to-end tokens/s = completion_tokens / request seconds. Decode tokens/s = "
             "(completion_tokens - 1) / (last - first output event seconds), an approximation "
             "when the server batches tokens. Reasoning is included in output timing and API completion usage. "
             "Missing API usage is N/A; chunks are never counted as tokens.", "",
             "Only complete requests enter averages; length-limited or failed requests are excluded. "
             "Prompt adherence is not graded. Raw responses and errors are in the JSON report.", "",
             "| Prompt | Complete / attempted | Avg tokens | Avg TTFT (s) | Avg elapsed (s) | Avg end-to-end tokens/s | Avg decode tokens/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in [*PROMPTS, "ALL"]:
        attempted = [r for r in rows if name == "ALL" or r["prompt"] == name]
        valid = [r for r in attempted if r["status"] == "ok"]
        cells = [number(average(valid, key)) for key in
                 ("completion_tokens", "ttft_s", "elapsed_s", "end_to_end_tps", "decode_tps")]
        lines.append(f"| {name} | {len(valid)} / {len(attempted)} | " + " | ".join(cells) + " |")
    valid = [r for r in rows if r["status"] == "ok" and r.get("completion_tokens") is not None]
    weighted = sum(r["completion_tokens"] for r in valid) / sum(r["elapsed_s"] for r in valid) if valid else None
    lines += ["", "ALL is the arithmetic mean over successful requests. Aggregate throughput "
              f"(total completion tokens / total elapsed seconds): **{number(weighted)} tokens/s**.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://0.0.0.0:8080/v1/chat/completions")
    parser.add_argument("--model", help="Defaults to the first model returned by /v1/models")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per prompt (default: 1; use 3+ for steadier averages)")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=600, help="Socket inactivity timeout in seconds")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("openai_speed_report"))
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("runs, max-tokens, and timeout must be positive")
    # Avoid environment proxies for the local inference endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if not args.model:
        try:
            models_url = args.endpoint.removesuffix("/chat/completions") + "/models"
            with request(opener, models_url, args.api_key, timeout=args.timeout) as response:
                args.model = json.load(response)["data"][0]["id"]
        except Exception as exc:
            parser.exit(1, f"Model discovery failed: {exc}. Supply --model explicitly.\n")
    rows = []
    for run in range(1, args.runs + 1):
        for name, prompt in PROMPTS.items():
            try:
                result = measure(opener, args, prompt)
            except (OSError, ValueError, RuntimeError) as exc:
                result = dict(status="error", error=str(exc))
            rows.append(dict(prompt=name, run=run, **result))
            save(args, rows)  # Keep partial results if interrupted.
            print(f"{run}/{args.runs} {name}: {result['status']}, "
                  f"end-to-end={number(result.get('end_to_end_tps'))} tokens/s", flush=True)
    print(f"Reports: {args.output.with_suffix('.md')} and .json")
    return int(any(r["status"] != "ok" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
