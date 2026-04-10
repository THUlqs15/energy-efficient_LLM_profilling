#!/usr/bin/env python3
"""
Batch latency profiling script for vLLM.
Sweeps GPU clock frequencies and collects diverse batch execution timing data.
Produces profiling_data.jsonl with per-batch records.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

# ── Environment setup (must be before vLLM imports) ──────────────────────────
os.environ["VLLM_PROFILING"] = "1"
os.environ["VLLM_PROFILING_WARMUP"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser(description="vLLM batch latency profiler")
    p.add_argument("--model-path", default="/home/ubuntu/lqs/L3")
    p.add_argument("--output", default="/home/ubuntu/lqs/profiling_data.jsonl")
    p.add_argument("--num-freqs", type=int, default=25,
                   help="Number of GPU frequency points to profile")
    p.add_argument("--warmup-batches", type=int, default=25,
                   help="Warmup batches per frequency (discarded in fitting)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── GPU frequency helpers ────────────────────────────────────────────────────

def get_supported_frequencies():
    """Parse all supported GPU graphics clock frequencies (MHz)."""
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
        capture_output=True, text=True, check=True,
    )
    freqs = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Graphics") and "MHz" in line:
            try:
                freq = int(line.split(":")[1].strip().split()[0])
                freqs.add(freq)
            except (IndexError, ValueError):
                pass
    return sorted(freqs)


def select_frequencies(all_freqs, n=25):
    """Select n evenly-spaced frequencies, avoiding the top ~10% of the range."""
    max_freq = max(all_freqs)
    cutoff = int(max_freq * 0.90)
    usable = [f for f in all_freqs if f <= cutoff]
    if len(usable) <= n:
        return usable
    indices = np.linspace(0, len(usable) - 1, n, dtype=int)
    return sorted(set(usable[i] for i in indices))


def lock_gpu_clock(freq_mhz):
    subprocess.run(
        ["sudo", "nvidia-smi", f"--lock-gpu-clocks={freq_mhz},{freq_mhz}"],
        check=True, capture_output=True,
    )
    time.sleep(0.5)  # let clock stabilise


def reset_gpu_clock():
    subprocess.run(
        ["sudo", "nvidia-smi", "--reset-gpu-clocks"],
        check=True, capture_output=True,
    )


def get_actual_gpu_freq():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


# ── Prompt generation ────────────────────────────────────────────────────────

def make_random_token_ids(length, vocab_size=151936):
    """Generate a list of random token IDs avoiding special tokens."""
    return [random.randint(1000, vocab_size - 1) for _ in range(length)]


# ── Engine creation ──────────────────────────────────────────────────────────

def create_engine(model_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    engine_args = EngineArgs(
        model=model_path,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    return engine


# ── Batch generation helpers ─────────────────────────────────────────────────

def drain_engine(engine):
    """Step until all requests finish. Returns number of steps taken."""
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    return steps


def add_requests(engine, n, l_q, max_tokens, req_id_counter):
    """Add n requests of prompt length l_q with given max_tokens."""
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    ids = []
    for _ in range(n):
        rid = f"req-{req_id_counter[0]}"
        req_id_counter[0] += 1
        prompt = {"prompt_token_ids": make_random_token_ids(l_q)}
        engine.add_request(rid, prompt, params)
        ids.append(rid)
    return ids


# ── Profiling campaigns ─────────────────────────────────────────────────────

def run_warmup(engine, n_batches, req_counter):
    """Run warmup batches to stabilise GPU state after clock change."""
    for _ in range(n_batches // 5 + 1):
        add_requests(engine, 4, l_q=64, max_tokens=3, req_id_counter=req_counter)
        drain_engine(engine)


def run_pure_prefill_sweep(engine, req_counter):
    """Phase A: pure prefill batches with varied l_q and batch size."""
    batch_sizes = [1, 2, 4, 8, 16]
    lq_values = [32, 64, 128, 256, 512, 1024, 2048]

    for bs in batch_sizes:
        for lq in lq_values:
            # Skip very large combos that might OOM
            if bs * lq > 16384:
                continue
            for _rep in range(2):
                add_requests(engine, bs, l_q=lq, max_tokens=2,
                             req_id_counter=req_counter)
                drain_engine(engine)


def run_decode_sweep(engine, req_counter):
    """Phase B: long decode sequences to get varied l_kv values."""
    configs = [
        # (batch_size, l_q, max_tokens)
        (1, 64, 300),
        (1, 256, 200),
        (2, 64, 200),
        (2, 128, 150),
        (4, 64, 150),
        (4, 128, 100),
        (8, 64, 100),
        (8, 128, 80),
        (16, 64, 80),
        (16, 128, 60),
        (32, 64, 50),
        (32, 128, 40),
        (64, 32, 30),
    ]
    for bs, lq, mt in configs:
        add_requests(engine, bs, l_q=lq, max_tokens=mt,
                     req_id_counter=req_counter)
        drain_engine(engine)


def run_mixed_injection(engine, req_counter, total_steps=300):
    """Phase C: continuous request injection for mixed prefill+decode batches."""
    lq_choices = [32, 64, 128, 256, 512, 1024]
    mt_choices = [30, 60, 100, 150]

    # Seed the engine with some initial decode requests
    add_requests(engine, 8, l_q=128, max_tokens=200,
                 req_id_counter=req_counter)

    steps_done = 0
    inject_interval = 5  # inject every N steps
    while steps_done < total_steps:
        # Inject new requests periodically
        if steps_done % inject_interval == 0 and steps_done > 0:
            n_inject = random.randint(1, 4)
            lq = random.choice(lq_choices)
            mt = random.choice(mt_choices)
            add_requests(engine, n_inject, l_q=lq, max_tokens=mt,
                         req_id_counter=req_counter)
            # Vary injection interval
            inject_interval = random.choice([3, 5, 7, 10])

        if not engine.has_unfinished_requests():
            # Re-seed if engine drained
            n_seed = random.randint(4, 16)
            lq = random.choice(lq_choices)
            mt = random.choice([100, 200, 300])
            add_requests(engine, n_seed, l_q=lq, max_tokens=mt,
                         req_id_counter=req_counter)

        engine.step()
        steps_done += 1

    # Drain remaining
    drain_engine(engine)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ["VLLM_PROFILING_FILE"] = args.output

    # Remove old output file
    if os.path.exists(args.output):
        os.remove(args.output)

    # Discover and select GPU frequencies
    all_freqs = get_supported_frequencies()
    print(f"Total supported frequencies: {len(all_freqs)} "
          f"(range {min(all_freqs)}-{max(all_freqs)} MHz)")

    frequencies = select_frequencies(all_freqs, n=args.num_freqs)
    print(f"Selected {len(frequencies)} frequencies for profiling:")
    print(f"  {frequencies}")

    # Create engine once (reuse across frequencies)
    print("\nInitialising vLLM engine …")
    engine = create_engine(args.model_path)
    print("Engine ready.\n")

    req_counter = [0]  # mutable counter for unique request IDs
    total_start = time.time()

    for fi, freq in enumerate(frequencies):
        freq_start = time.time()
        print(f"[{fi+1}/{len(frequencies)}] Profiling at {freq} MHz …")

        # Lock GPU clock
        lock_gpu_clock(freq)
        actual = get_actual_gpu_freq()
        print(f"  Locked GPU clock: target={freq} MHz, actual={actual} MHz")

        # ── Warmup ────────────────────────────────────────────────────────
        run_warmup(engine, args.warmup_batches, req_counter)

        # ── Phase A: Pure prefill sweep ───────────────────────────────────
        print("  Phase A: pure prefill sweep …")
        run_pure_prefill_sweep(engine, req_counter)

        # ── Phase B: Decode sweep ─────────────────────────────────────────
        print("  Phase B: decode sweep …")
        run_decode_sweep(engine, req_counter)

        # ── Phase C: Mixed injection ──────────────────────────────────────
        print("  Phase C: mixed batch injection …")
        run_mixed_injection(engine, req_counter, total_steps=300)

        # Reset GPU clock
        reset_gpu_clock()

        elapsed = time.time() - freq_start
        # Count records for this frequency
        if os.path.exists(args.output):
            with open(args.output) as f:
                n_records = sum(1 for _ in f)
        else:
            n_records = 0
        print(f"  Done in {elapsed:.1f}s  (total records so far: {n_records})")

    total_elapsed = time.time() - total_start

    # Final summary
    if os.path.exists(args.output):
        with open(args.output) as f:
            records = [json.loads(line) for line in f]
        freqs_found = sorted(set(int(round(r["gpu_freq_mhz"])) for r in records))
        print(f"\n{'='*60}")
        print(f"Profiling complete in {total_elapsed:.1f}s")
        print(f"Total batch records: {len(records)}")
        print(f"Distinct frequencies: {len(freqs_found)}")
        print(f"Output file: {args.output}")
    else:
        print("WARNING: No output file produced!")

    # Safety: always reset GPU clocks
    reset_gpu_clock()
    print("GPU clocks reset. Done.")


if __name__ == "__main__":
    main()
