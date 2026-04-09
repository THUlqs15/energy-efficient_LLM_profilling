#!/usr/bin/env python3
"""
LLMEngine-based Batch Latency Profiling Script for vLLM.

This script uses LLMEngine directly with step() to capture both prefill
and decode batches. Unlike the high-level generate() API, this approach
gives us visibility into each scheduler step including decode iterations.

Usage:
    conda activate myvllm
    PYTHONPATH=/home/ubuntu/lqs/vllm python profiling_engine_script.py \
        --model-path /home/ubuntu/lqs/L3 \
        --output /home/ubuntu/lqs/profiling_data.jsonl
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def query_supported_gpu_clocks():
    """Query all supported GPU graphics clock frequencies."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
            capture_output=True, text=True, check=True,
        )
        freqs = []
        for line in result.stdout.split("\n"):
            if "Graphics" in line and "MHz" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    freq_str = parts[1].strip().split()[0]
                    try:
                        freqs.append(int(freq_str))
                    except ValueError:
                        pass
        return sorted(set(freqs), reverse=True)
    except Exception as e:
        print(f"Error querying GPU clocks: {e}")
        return []


def select_representative_frequencies(all_freqs, num_freqs=8):
    """Select representative frequencies spanning the full range."""
    if len(all_freqs) <= num_freqs:
        return all_freqs
    percentiles = [0, 15, 30, 45, 60, 75, 90, 100]
    selected_indices = [int(len(all_freqs) * p / 100) for p in percentiles]
    selected_indices = [min(i, len(all_freqs) - 1) for i in selected_indices]
    seen = set()
    result = []
    for freq in [all_freqs[i] for i in selected_indices]:
        if freq not in seen:
            seen.add(freq)
            result.append(freq)
    return result


def lock_gpu_clock(freq_mhz):
    """Lock GPU clock to specified frequency using sudo."""
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "--lock-gpu-clocks", f"{freq_mhz},{freq_mhz}"],
            check=True, capture_output=True,
        )
        print(f"  Locked GPU clock to {freq_mhz} MHz")
        time.sleep(2)  # Give GPU time to stabilize

        # Verify the lock
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        actual_freq = int(result.stdout.strip())
        if abs(actual_freq - freq_mhz) > 100:
            print(f"  Warning: Requested {freq_mhz} MHz but got {actual_freq} MHz")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error locking GPU clock: {e}")
        return False


def reset_gpu_clock():
    """Reset GPU clock to default using sudo."""
    try:
        subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"],
                      check=True, capture_output=True)
        print("  Reset GPU clock to default")
    except Exception as e:
        print(f"  Error resetting GPU clock: {e}")


def generate_prompts_for_workload(workload_type, num_requests, prompt_lengths, max_tokens):
    """
    Generate prompts for different workload types.

    Args:
        workload_type: "prefill_heavy", "decode_heavy", or "mixed"
        num_requests: Number of requests to generate
        prompt_lengths: List of prompt lengths to sample from
        max_tokens: Max tokens to generate per request

    Returns:
        List of (prompt, max_tokens) tuples
    """
    prompts = []
    rng = np.random.default_rng(42)

    for i in range(num_requests):
        prompt_len = int(rng.choice(prompt_lengths))
        # Create prompt with approximately prompt_len tokens
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog."]
        num_repeats = max(1, prompt_len // len(words))
        prompt = " ".join(words * num_repeats)

        if workload_type == "prefill_heavy":
            # Long prompts, short generation
            tokens = int(rng.integers(1, 10))
        elif workload_type == "decode_heavy":
            # Short prompts, long generation
            tokens = max_tokens
        else:  # mixed
            tokens = int(rng.integers(10, max_tokens))

        prompts.append((prompt, tokens))

    return prompts


def profile_with_engine(
    model_path,
    output_file,
    num_requests_per_batch=8,
    prompt_lengths=[64, 128, 256, 512],
    max_tokens_range=[20, 50, 100],
    num_iterations=100,
    warmup_iterations=20,
):
    """
    Profile using LLMEngine directly with step() to capture decode batches.
    """
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm import profiling_logger

    print(f"  Initializing LLMEngine with model: {model_path}")

    # Create engine arguments
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=64,
        enforce_eager=True,  # Disable CUDA graphs for consistent timing
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=True,
    )

    # Create the engine
    engine = LLMEngine.from_engine_args(engine_args)

    # Reset profiling counter for this frequency
    profiling_logger.reset_batch_counter()

    rng = np.random.default_rng(42)
    total_steps = 0
    request_id_counter = 0

    print(f"  Running {num_iterations} iterations (+ {warmup_iterations} warmup)")

    for iteration in range(num_iterations + warmup_iterations):
        is_warmup = iteration < warmup_iterations

        # Vary workload type
        if iteration % 3 == 0:
            workload_type = "prefill_heavy"
        elif iteration % 3 == 1:
            workload_type = "decode_heavy"
        else:
            workload_type = "mixed"

        # Generate requests for this iteration
        num_reqs = int(rng.integers(1, num_requests_per_batch + 1))
        max_tokens = int(rng.choice(max_tokens_range))
        prompts = generate_prompts_for_workload(
            workload_type, num_reqs, prompt_lengths, max_tokens
        )

        # Add requests to engine
        for prompt, tokens in prompts:
            request_id = f"req-{request_id_counter}"
            request_id_counter += 1

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=tokens,
                ignore_eos=True,
            )

            try:
                engine.add_request(request_id, prompt, sampling_params)
            except Exception as e:
                print(f"  Warning: Failed to add request: {e}")
                continue

        # Run engine steps until all requests complete
        steps_this_iteration = 0
        while engine.has_unfinished_requests():
            try:
                outputs = engine.step()
                steps_this_iteration += 1
                total_steps += 1

                if total_steps % 500 == 0:
                    print(f"    Total steps: {total_steps}")

            except Exception as e:
                print(f"  Warning: step() failed: {e}")
                break

        if not is_warmup and (iteration - warmup_iterations + 1) % 20 == 0:
            print(f"    Iteration {iteration - warmup_iterations + 1}/{num_iterations}, "
                  f"steps this iter: {steps_this_iteration}")

    print(f"  Completed {total_steps} total steps")

    # Cleanup
    del engine
    torch.cuda.empty_cache()

    return total_steps


def profile_at_frequency(freq_mhz, model_path, output_file, num_iterations=100):
    """Profile at a specific GPU frequency."""
    print(f"\n=== Profiling at {freq_mhz} MHz ===")

    if not lock_gpu_clock(freq_mhz):
        print(f"  Skipping frequency {freq_mhz} MHz due to clock lock failure")
        return 0

    # Set profiling environment variables
    os.environ["VLLM_PROFILING"] = "1"
    os.environ["VLLM_PROFILING_FILE"] = output_file
    os.environ["VLLM_PROFILING_WARMUP"] = "30"  # More warmup for stability

    try:
        steps = profile_with_engine(
            model_path=model_path,
            output_file=output_file,
            num_requests_per_batch=8,
            prompt_lengths=[64, 128, 256, 512, 1024],
            max_tokens_range=[20, 50, 100, 150],
            num_iterations=num_iterations,
            warmup_iterations=30,
        )
        return steps

    except Exception as e:
        print(f"  Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 0

    finally:
        reset_gpu_clock()


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM batch latency using LLMEngine (captures decode batches)"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the LLM model"
    )
    parser.add_argument(
        "--output", type=str, default="/home/ubuntu/lqs/profiling_data.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--num-frequencies", type=int, default=6,
        help="Number of GPU frequencies to profile"
    )
    parser.add_argument(
        "--iterations-per-freq", type=int, default=80,
        help="Number of workload iterations per frequency"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("vLLM Batch Latency Profiling (LLMEngine-based, captures decode)")
    print("=" * 80)

    # Check GPU clock control
    print("\nChecking GPU clock control permissions...")
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "--lock-gpu-clocks", "210,210"],
            check=True, capture_output=True, timeout=5,
        )
        subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"],
                      check=True, capture_output=True)
        print("✓ GPU clock control is available")
    except Exception as e:
        print(f"✗ GPU clock control failed: {e}")
        sys.exit(1)

    # Query supported frequencies
    print("\nQuerying supported GPU frequencies...")
    all_freqs = query_supported_gpu_clocks()
    if not all_freqs:
        print("Error: Could not query GPU frequencies")
        sys.exit(1)

    print(f"  Found {len(all_freqs)} supported frequencies")
    print(f"  Range: {min(all_freqs)} - {max(all_freqs)} MHz")

    # Select representative frequencies
    selected_freqs = select_representative_frequencies(all_freqs, args.num_frequencies)
    print(f"  Selected {len(selected_freqs)} frequencies for profiling:")
    for freq in selected_freqs:
        print(f"    {freq} MHz")

    # Clear output file
    output_path = Path(args.output)
    if output_path.exists():
        print(f"\nRemoving existing profiling data: {args.output}")
        output_path.unlink()

    # Profile at each frequency
    print("\nStarting profiling...")
    print(f"  Output file: {args.output}")
    print(f"  Iterations per frequency: {args.iterations_per_freq}")

    total_steps = 0
    for freq_idx, freq_mhz in enumerate(selected_freqs):
        print(f"\nFrequency {freq_idx + 1}/{len(selected_freqs)}: {freq_mhz} MHz")

        steps = profile_at_frequency(
            freq_mhz=freq_mhz,
            model_path=args.model_path,
            output_file=args.output,
            num_iterations=args.iterations_per_freq,
        )
        total_steps += steps

    # Final cleanup
    reset_gpu_clock()

    # Count data
    if output_path.exists():
        with open(output_path) as f:
            num_records = sum(1 for _ in f)
    else:
        num_records = 0

    print("\n" + "=" * 80)
    print("Profiling complete!")
    print(f"  Total engine steps: {total_steps}")
    print(f"  Total batch records: {num_records}")
    print(f"  Output file: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
