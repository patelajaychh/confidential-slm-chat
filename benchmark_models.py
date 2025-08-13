#!/usr/bin/env python3
"""
Benchmark Hugging Face models in Confidential VM vs Non-Confidential VM.
Measures latency (p50/p90/p95/p99), throughput, memory, and cold-start times.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import random
from statistics import mean, stdev
from subprocess import run
import time

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ----------------
MODELS = [
    "distilgpt2",
    "sshleifer/tiny-gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5"
]

TOKEN_LENGTHS = [32, 128]
BATCH_SIZES = [1, 8]
CONCURRENCY_LEVELS = [1, 8]
WARMUP_REQUESTS = 5
TEST_REQUESTS = 20
OUTPUT_CSV = "benchmark_results.csv"
USE_PERF = False  # Set to True to collect perf stat metrics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------

def collect_perf_metrics(duration=10):
    """Run perf stat for the given duration."""
    if not USE_PERF:
        return {}
    cmd = [
        "perf", "stat", "-a",
        "-e", "cycles,instructions,cache-misses,context-switches,minor-faults,major-faults",
        "sleep", str(duration)
    ]
    result = run(cmd, capture_output=True, text=True)
    return {"perf_output": result.stderr}

def benchmark_model(model_name, token_length, batch_size, concurrency):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    prompt = "India was known as Golden Bird before"  
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to(DEVICE)

    # Measure cold start (model load already done, here just generation warm-up)
    # Warm-up
    for _ in range(WARMUP_REQUESTS):
        _ = model.generate(**inputs, max_new_tokens=token_length)

    # Benchmark steady-state
    latencies = []
    start_time = time.time()

    def run_inference():
        t0 = time.time()
        _ = model.generate(**inputs, max_new_tokens=token_length)
        latencies.append(time.time() - t0)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(TEST_REQUESTS):
            executor.submit(run_inference)

    total_time = time.time() - start_time
    tokens_generated = TEST_REQUESTS * token_length * batch_size
    throughput_tok_s = tokens_generated / total_time

    # Memory usage
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    # Latency stats
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    result = {
        "model": model_name,
        "token_length": token_length,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "p50_latency_s": round(p50, 4),
        "p90_latency_s": round(p90, 4),
        "p95_latency_s": round(p95, 4),
        "p99_latency_s": round(p99, 4),
        "throughput_tokens_per_s": round(throughput_tok_s, 2),
        "memory_mb": round(mem_mb, 2),
        "total_time_s": round(total_time, 4)
    }

    if USE_PERF:
        result.update(collect_perf_metrics(duration=3))

    return result

def main():
    results = []
    for model in MODELS:
        for tok_len in TOKEN_LENGTHS:
            for bs in BATCH_SIZES:
                for conc in CONCURRENCY_LEVELS:
                    print(f"Benchmarking {model} | tokens={tok_len} | batch={bs} | conc={conc}")
                    res = benchmark_model(model, tok_len, bs, conc)
                    results.append(res)
                    df = pd.DataFrame(results)
                    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()