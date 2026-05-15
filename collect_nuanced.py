#!/usr/bin/env python3
"""
Collect Study 2 responses: 40 nuanced/contested items, Conditions A & B only.
40 items × 2 conditions × 3 models × 3 runs = 720 responses.
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
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "gemini-2.0-flash": "google/gemini-2.0-flash",
}

RUNS = 3
MAX_TOKENS = 1200
TEMPERATURE = 0
NUM_THREADS = 10
OUTPUT_CSV = "responses_nuanced.csv"

CSV_FIELDS = [
    "item_id", "domain", "misconception", "condition", "model", "run",
    "prompt", "response", "finish_reason", "prompt_tokens",
    "completion_tokens", "total_tokens", "is_truncated", "is_empty",
    "is_refusal", "error", "timestamp",
]

csv_lock = threading.Lock()


def load_items(path):
    with open(path) as f:
        return json.load(f)


def load_completed(csv_path):
    completed = set()
    if not Path(csv_path).exists():
        return completed
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            completed.add((int(row["item_id"]), row["condition"], row["model"], int(row["run"])))
    return completed


def ensure_csv(path):
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def write_row(path, row):
    with csv_lock:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def call_api(api_key, model_id, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sycophancy-research.local",
        "X-Title": "Sycophancy Research Study 2",
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
            p in content.lower()
            for p in ["i cannot", "i can't", "i'm unable", "i must decline"]
        ),
        "error": "",
    }


def process_task(api_key, item, condition, model_name, run, csv_path):
    prompt_key = f"condition_{condition.lower()}"
    prompt = item[prompt_key]
    try:
        result = call_api(api_key, MODELS[model_name], prompt)
    except Exception as e:
        result = {
            "response": "", "finish_reason": "", "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "is_truncated": False,
            "is_empty": True, "is_refusal": False, "error": str(e),
        }

    row = {
        "item_id": item["id"], "domain": item["domain"],
        "misconception": item["misconception"], "condition": condition,
        "model": model_name, "run": run, "prompt": prompt,
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
        sys.exit(1)

    script_dir = Path(__file__).parent
    items = load_items(str(script_dir / "items_nuanced.json"))
    csv_path = str(script_dir / OUTPUT_CSV)

    ensure_csv(csv_path)
    completed = load_completed(csv_path)

    tasks = []
    for item in items:
        for cond in ["A", "B"]:
            for model in MODELS:
                for run in range(1, RUNS + 1):
                    if (item["id"], cond, model, run) not in completed:
                        tasks.append((item, cond, model, run))

    total = len(tasks)
    expected = len(items) * 2 * len(MODELS) * RUNS
    print(f"Items: {len(items)} | Tasks: {total}/{expected}")

    if total == 0:
        print("All done!")
        return

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(process_task, api_key, item, cond, model, run, csv_path): (item["id"], cond, model, run)
            for item, cond, model, run in tasks
        }
        for future in as_completed(futures):
            done += 1
            try:
                row = future.result()
                if row.get("error"):
                    errors += 1
            except Exception:
                errors += 1
            if done % 30 == 0 or done == total:
                print(f"[{done}/{total}] errors: {errors}")

    print(f"\nDone. {done} collected, {errors} errors. → {csv_path}")


if __name__ == "__main__":
    main()
