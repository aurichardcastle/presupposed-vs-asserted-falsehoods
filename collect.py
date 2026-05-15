#!/usr/bin/env python3
"""
Data collection script for sycophancy research.
Sends 120 prompts × 3 models × 3 runs = 1,080 responses via OpenRouter API.
Supports resume: skips already-collected rows found in the output CSV.
"""

import json
import os
import sys
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}

RUNS_PER_COMBO = 3
MAX_TOKENS = 1200
TEMPERATURE = 0
NUM_THREADS = 10
OUTPUT_CSV = "responses.csv"

CSV_FIELDS = [
    "item_id",
    "domain",
    "misconception",
    "condition",
    "model",
    "run",
    "prompt",
    "response",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "is_truncated",
    "is_empty",
    "is_refusal",
    "error",
    "timestamp",
]

csv_lock = threading.Lock()


def load_items(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_completed(csv_path: str) -> set[tuple]:
    completed = set()
    if not Path(csv_path).exists():
        return completed
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error"):
                continue
            key = (
                int(row["item_id"]),
                row["condition"],
                row["model"],
                int(row["run"]),
            )
            completed.add(key)
    return completed


def write_row(csv_path: str, row: dict, write_header: bool = False):
    with csv_lock:
        mode = "a" if Path(csv_path).exists() and not write_header else "w"
        file_exists = Path(csv_path).exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists or write_header:
                writer.writeheader()
            writer.writerow(row)


def ensure_csv_header(csv_path: str):
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def call_openrouter(api_key: str, model_id: str, prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sycophancy-research.local",
        "X-Title": "Sycophancy Research",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "") or ""
    finish_reason = choice.get("finish_reason", "")
    usage = data.get("usage", {})
    refusal = message.get("refusal")

    return {
        "response": content,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "is_truncated": finish_reason == "length",
        "is_empty": len(content.strip()) == 0,
        "is_refusal": bool(refusal) or any(
            phrase in content.lower()
            for phrase in ["i cannot", "i can't", "i'm unable", "i must decline"]
        ),
        "error": "",
    }


def process_task(
    api_key: str,
    item: dict,
    condition: str,
    model_name: str,
    run: int,
    csv_path: str,
):
    prompt_key = f"condition_{condition.lower()}"
    prompt_text = item[prompt_key]
    model_id = MODELS[model_name]

    try:
        result = call_openrouter(api_key, model_id, prompt_text)
    except Exception as e:
        result = {
            "response": "",
            "finish_reason": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "is_truncated": False,
            "is_empty": True,
            "is_refusal": False,
            "error": str(e),
        }

    row = {
        "item_id": item["id"],
        "domain": item["domain"],
        "misconception": item["misconception"],
        "condition": condition,
        "model": model_name,
        "run": run,
        "prompt": prompt_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **result,
    }

    write_row(csv_path, row)
    return row


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ").strip()
    if not api_key:
        print("Error: No API key provided.")
        sys.exit(1)

    script_dir = Path(__file__).parent
    items_path = script_dir / "items.json"
    csv_path = script_dir / OUTPUT_CSV

    items = load_items(str(items_path))
    print(f"Loaded {len(items)} items")

    completed = load_completed(str(csv_path))
    print(f"Found {len(completed)} already-completed responses")

    ensure_csv_header(str(csv_path))

    tasks = []
    conditions = ["A", "B", "C"]
    for item in items:
        for condition in conditions:
            for model_name in MODELS:
                for run in range(1, RUNS_PER_COMBO + 1):
                    key = (item["id"], condition, model_name, run)
                    if key in completed:
                        continue
                    tasks.append((item, condition, model_name, run))

    total = len(tasks)
    print(f"Tasks to run: {total} (of {len(items) * 3 * len(MODELS) * RUNS_PER_COMBO} total)")

    if total == 0:
        print("All responses already collected!")
        return

    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(
                process_task,
                api_key,
                item,
                condition,
                model_name,
                run,
                str(csv_path),
            ): (item["id"], condition, model_name, run)
            for item, condition, model_name, run in tasks
        }

        for future in as_completed(futures):
            key = futures[future]
            done += 1
            try:
                row = future.result()
                status = "OK"
                if row.get("error"):
                    status = f"ERROR: {row['error'][:60]}"
                    errors += 1
                elif row.get("is_truncated"):
                    status = "TRUNCATED"
                elif row.get("is_refusal"):
                    status = "REFUSAL"
                elif row.get("is_empty"):
                    status = "EMPTY"
            except Exception as e:
                status = f"EXCEPTION: {e}"
                errors += 1

            item_id, cond, model, run = key
            print(
                f"[{done}/{total}] item={item_id} cond={cond} model={model} run={run} → {status}"
            )

    print(f"\nDone. {done} responses collected, {errors} errors.")
    print(f"Output: {csv_path}")


if __name__ == "__main__":
    main()
