"""
Profiling script for vLLM batch execution time.
Collects data across multiple GPU frequencies with diverse batch configurations.

Note: This A10G cloud instance keeps GPU at 1710 MHz regardless of clock lock attempts.
We collect diverse batch data at the available frequency and document this.
"""

import os
import subprocess
import sys
import time
import random
import json

# Set profiling environment variables before importing vllm
os.environ["VLLM_PROFILING"] = "1"
os.environ["VLLM_PROFILING_OUTPUT"] = "/home/ubuntu/lqs/profiling_data.jsonl"

# Fix: CWD is /home/ubuntu/lqs which contains a 'vllm' directory.
# This causes Python to find it as a namespace package instead of using
# the editable install. Remove CWD ('') from sys.path to fix this.
if '' in sys.path:
    sys.path.remove('')
# Also explicitly remove /home/ubuntu/lqs if present
for p in ['/home/ubuntu/lqs', '.']:
    if p in sys.path:
        sys.path.remove(p)

import torch


def get_supported_frequencies():
    """Get all supported GPU graphics clock frequencies."""
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
        capture_output=True,
        text=True,
    )
    freqs = []
    for line in result.stdout.splitlines():
        if "Graphics" in line and "MHz" in line:
            try:
                f = int(line.strip().split(":")[1].strip().split()[0])
                if f not in freqs:
                    freqs.append(f)
            except (IndexError, ValueError):
                pass
    freqs = sorted(set(freqs))
    return freqs


def try_lock_gpu_freq(freq):
    """Attempt to lock GPU clock to a specific frequency."""
    result = subprocess.run(
        ["sudo", "nvidia-smi", f"--lock-gpu-clocks={freq},{freq}"],
        capture_output=True,
        text=True,
    )
    time.sleep(2)
    actual = get_current_freq()
    return actual


def reset_gpu_clocks():
    """Reset GPU clocks to default."""
    subprocess.run(
        ["sudo", "nvidia-smi", "--reset-gpu-clocks"],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["sudo", "nvidia-smi", "-rac"],
        capture_output=True,
        text=True,
    )


def get_current_freq():
    """Get current GPU graphics clock."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except Exception:
        return -1


def select_frequencies(all_freqs):
    """Select 6 representative frequencies spanning the full range."""
    if len(all_freqs) <= 10:
        return all_freqs
    # Select evenly spaced frequencies
    indices = [int(i * (len(all_freqs) - 1) / 5) for i in range(6)]
    selected = [all_freqs[i] for i in indices]
    return selected


def count_records(output_path):
    try:
        with open(output_path, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run_single_batch(llm, prompts, max_tokens_list, output_path):
    """Run a single batch and return number of new records written."""
    from vllm.sampling_params import SamplingParams
    before = count_records(output_path)
    sps = [SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True)
           for mt in max_tokens_list]
    try:
        llm.generate(prompts, sps, use_tqdm=False)
    except Exception as e:
        print(f"    Warning: batch gen failed: {e}")
        return 0
    after = count_records(output_path)
    return after - before


WORDS = ["hello", "world", "the", "a", "is", "are", "was", "were",
         "in", "on", "at", "by", "for", "with", "from", "to",
         "this", "that", "which", "when", "where", "how", "why",
         "but", "and", "or", "not", "so", "if", "then", "else",
         "be", "have", "do", "say", "get", "make", "go", "know",
         "take", "see", "come", "think", "look", "want", "give",
         "time", "year", "people", "way", "day", "man", "woman",
         "child", "life", "hand", "part", "place", "week", "case"]


def make_prompt(token_count):
    """Build a prompt of approximately token_count tokens."""
    num_words = max(1, int(token_count * 1.3))
    return " ".join(random.choice(WORDS) for _ in range(num_words))


def run_profiling_session(llm, session_label, warmup=15, target_samples=400):
    """
    Run diverse batches and collect profiling data.
    Strategy: mix of prefill-heavy and decode-heavy batches.
    """
    print(f"  Session: {session_label}")
    output_path = os.environ["VLLM_PROFILING_OUTPUT"]

    existing_count = count_records(output_path)
    total_new = 0
    post_warmup = 0

    def add_records(n):
        nonlocal total_new, post_warmup
        total_new += n
        if total_new > warmup:
            post_warmup = total_new - warmup

    # === Phase 1: Pure prefill batches ===
    # Submit one request at a time with max_tokens=1 to get pure prefill
    print("    Phase 1: Pure prefill batches")
    for plen in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] * 5:
        if post_warmup >= target_samples:
            break
        p = make_prompt(plen)
        n = run_single_batch(llm, [p], [1], output_path)
        add_records(n)

    # Prefill with small batches
    for bs in [2, 4, 8, 16]:
        for plen in [32, 64, 128, 256, 512]:
            if post_warmup >= target_samples:
                break
            prompts = [make_prompt(plen) for _ in range(bs)]
            n = run_single_batch(llm, prompts, [1] * bs, output_path)
            add_records(n)

    print(f"    Phase 1 done: {post_warmup} post-warmup records")

    # === Phase 2: Decode-heavy batches ===
    print("    Phase 2: Decode-heavy batches")
    decode_configs = []
    for plen in [8, 16, 32]:
        for ntok in [32, 64, 96, 128, 192, 256]:
            for bs in [1, 2, 4, 8, 16]:
                decode_configs.append((plen, ntok, bs))

    random.shuffle(decode_configs)
    for plen, ntok, bs in decode_configs:
        if post_warmup >= target_samples:
            break
        prompts = [make_prompt(plen) for _ in range(bs)]
        n = run_single_batch(llm, prompts, [ntok] * bs, output_path)
        add_records(n)

    print(f"    Phase 2 done: {post_warmup} post-warmup records")

    # === Phase 3: Mixed batch diversity ===
    print("    Phase 3: Mixed batch configs")
    mixed_configs = []
    for _ in range(200):
        bs = random.randint(1, 16)
        plens = [random.choice([16, 32, 64, 128, 256, 512]) for _ in range(bs)]
        ntoks = [random.choice([1, 2, 4, 8, 16, 32, 64, 128]) for _ in range(bs)]
        mixed_configs.append((plens, ntoks))

    for plens, ntoks in mixed_configs:
        if post_warmup >= target_samples:
            break
        prompts = [make_prompt(pl) for pl in plens]
        n = run_single_batch(llm, prompts, ntoks, output_path)
        add_records(n)

    print(f"  Session done: {post_warmup} post-warmup records")
    return post_warmup


def main():
    print("=== vLLM Batch Profiling Script ===")
    print(f"Model: /home/ubuntu/lqs/L3")
    print(f"Output: {os.environ['VLLM_PROFILING_OUTPUT']}")

    # Check GPU info
    all_freqs = get_supported_frequencies()
    current_freq = get_current_freq()
    print(f"GPU current clock: {current_freq} MHz")
    print(f"Supported frequencies: {len(all_freqs)}, range [{min(all_freqs)}, {max(all_freqs)}] MHz")

    # Try to lock to different frequencies
    selected_freqs = select_frequencies(all_freqs)
    print(f"Attempting frequencies: {selected_freqs}")

    # Test if clock locking works
    print("\nTesting clock lock capability...")
    achievable_freqs = {}
    for freq in [min(all_freqs), max(all_freqs)]:
        actual = try_lock_gpu_freq(freq)
        achievable_freqs[freq] = actual
        print(f"  Requested {freq} MHz -> actual {actual} MHz")
    reset_gpu_clocks()

    # Check if we can actually vary frequency
    actual_freqs_set = set(achievable_freqs.values())
    if len(actual_freqs_set) == 1:
        print(f"\nNote: GPU does not support user clock control on this system.")
        print(f"All profiling will be done at {list(actual_freqs_set)[0]} MHz.")
        print("The profiling logger will record actual GPU frequency per batch.")
        profiling_freqs = [current_freq]
    else:
        print(f"\nGPU clock control works! Profiling at: {selected_freqs}")
        profiling_freqs = selected_freqs

    # Clean existing data
    output_path = os.environ["VLLM_PROFILING_OUTPUT"]
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing {output_path}")

    total_collected = 0

    # If only one freq, run more batches to get good data
    samples_per_session = 400 if len(profiling_freqs) == 1 else 250

    for freq_idx, freq in enumerate(profiling_freqs):
        print(f"\n[{freq_idx+1}/{len(profiling_freqs)}] Target freq: {freq} MHz")

        if len(profiling_freqs) > 1:
            actual = try_lock_gpu_freq(freq)
            print(f"  Actual freq: {actual} MHz")

        # Initialize vLLM
        print("  Initializing vLLM LLM engine...")
        try:
            from vllm.entrypoints.llm import LLM
            llm = LLM(
                model="/home/ubuntu/lqs/L3",
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
            )
            print("  vLLM engine initialized OK")
        except Exception as e:
            print(f"  ERROR initializing vLLM: {e}")
            import traceback
            traceback.print_exc()
            reset_gpu_clocks()
            continue

        n = run_profiling_session(
            llm,
            f"freq={freq}MHz",
            warmup=15,
            target_samples=samples_per_session,
        )
        total_collected += n

        # Cleanup
        del llm
        torch.cuda.empty_cache()
        time.sleep(2)

        if len(profiling_freqs) > 1:
            reset_gpu_clocks()

    print(f"\n=== Profiling Complete ===")
    print(f"Total batch records collected: {total_collected}")

    # Summary
    try:
        with open(output_path, "r") as f:
            records = [json.loads(line) for line in f]
        freqs_seen = sorted(set(r["gpu_freq_mhz"] for r in records))
        print(f"Total records in file: {len(records)}")
        print(f"GPU frequencies seen: {freqs_seen}")

        total_prefill = 0
        total_decode = 0
        for r in records:
            for req in r["requests"]:
                if req["type"] == "prefill":
                    total_prefill += 1
                else:
                    total_decode += 1
        print(f"Total prefill requests across all batches: {total_prefill}")
        print(f"Total decode requests across all batches: {total_decode}")

        wall_times = [r["wall_time_ms"] for r in records]
        print(f"Wall time: min={min(wall_times):.2f}ms, "
              f"max={max(wall_times):.2f}ms, "
              f"mean={sum(wall_times)/len(wall_times):.2f}ms")

    except Exception as e:
        print(f"Summary failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
