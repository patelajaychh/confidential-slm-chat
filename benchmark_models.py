"""
# Usage 
python3 benchmark_models.py
"""

import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODELS = [
    "distilgpt2",
    "sshleifer/tiny-gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5"
]

PROMPTS = [
    "The Prime Minister of India is",
    "The capital of France is",
    "The weather today is",
    "I love to eat",
    "The best way to learn programming is",
    "When I wake up in the morning, I",
    "The most important thing in life is",
    "My favorite movie is",
    "The internet was invented by",
    "The future of artificial intelligence"
]
MAX_NEW_TOKENS = 20
OUTPUT_FILE = "benchmark_results.json"  # Use JSON lines for appending

def benchmark_model(model_name):
    print(f"\nBenchmarking {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cpu")
    model.eval()

    results = []
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            start_mem = psutil.Process().memory_info().rss
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = end_time - start_time
        memory_used = (end_mem - start_mem) / (1024 ** 2)
        input_len = inputs["input_ids"].shape[1]
        output_len = outputs.shape[1]
        tokens_generated = output_len - input_len
        tokens_per_sec = tokens_generated / time_taken if time_taken > 0 else 0
        print(f"Output: {gen_text}")
        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Memory used: {memory_used:.3f} MB")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        results.append({
            "prompt": prompt,
            "output": gen_text,
            "time_s": time_taken,
            "memory_mb": memory_used,
            "tokens": tokens_generated,
            "throughput_tok_per_s": tokens_per_sec
        })
    # Calculate averages
    avg_time = sum(r["time_s"] for r in results) / len(results)
    avg_memory = sum(r["memory_mb"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_throughput = sum(r["throughput_tok_per_s"] for r in results) / len(results)
    print(f"\nAverages for {model_name}:")
    print(f"Time: {avg_time:.4f} s, Memory: {avg_memory:.2f} MB, Tokens: {avg_tokens:.2f}, Throughput: {avg_throughput:.2f} tok/s")
    return {
        "model": model_name,
        "results": results,
        "avg_time_s": avg_time,
        "avg_memory_mb": avg_memory,
        "avg_tokens": avg_tokens,
        "avg_throughput_tok_per_s": avg_throughput
    }

def print_summary_from_file():
    print("\nSummary (Averages):")
    print(f"{'Model':30} {'Time (s)':>10} {'Memory (MB)':>12} {'Tokens':>8} {'Tok/s':>10}")
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            r = json.loads(line)
            print(f"{r['model']:30} {r['avg_time_s']:10.4f} {r['avg_memory_mb']:12.2f} {r['avg_tokens']:8.2f} {r['avg_throughput_tok_per_s']:10.2f}")

def main():
    # Clear the output file at the start
    with open(OUTPUT_FILE, "w") as f:
        pass
    for model_name in MODELS:
        try:
            result = benchmark_model(model_name)
            # Write result to file after each model
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    print(f"\nDetailed results written to {OUTPUT_FILE}")
    print_summary_from_file()

if __name__ == "__main__":
    main() 