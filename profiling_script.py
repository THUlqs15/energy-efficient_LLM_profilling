#!/usr/bin/env python3
"""
Batch Latency Profiling Script for vLLM.

This script profiles vLLM batch execution latency across different GPU frequencies
and batch configurations to collect data for fitting a latency model.

Usage:
    python profiling_script.py --model-path /path/to/model --output profiling_data.jsonl

Environment:
    - Must run with VLLM_PROFILING=1 environment variable
    - Requires root/sudo for GPU clock locking (nvidia-smi --lock-gpu-clocks)
    - Conda environment: myvllm
"""

import argparse
import itertools
import json
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
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse graphics clock frequencies
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
        # Return unique frequencies sorted in descending order
        return sorted(set(freqs), reverse=True)
    except Exception as e:
        print(f"Error querying GPU clocks: {e}")
        return []


def select_representative_frequencies(all_freqs, num_freqs=8):
    """
    Select a representative subset of frequencies spanning the full range.
    Uses percentiles: min, 10%, 25%, 40%, 60%, 75%, 90%, max
    """
    if len(all_freqs) <= num_freqs:
        return all_freqs

    # Use percentiles to get good coverage
    percentiles = [0, 10, 25, 40, 60, 75, 90, 100]
    selected_indices = [
        int(len(all_freqs) * p / 100) for p in percentiles
    ]
    # Ensure max index is valid
    selected_indices = [min(i, len(all_freqs) - 1) for i in selected_indices]
    selected_freqs = [all_freqs[i] for i in selected_indices]

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for freq in selected_freqs:
        if freq not in seen:
            seen.add(freq)
            result.append(freq)

    return result


def lock_gpu_clock(freq_mhz):
    """Lock GPU graphics clock to specified frequency using sudo."""
    try:
        # Lock to a single frequency - use sudo
        cmd = ["sudo", "nvidia-smi", "--lock-gpu-clocks", f"{freq_mhz},{freq_mhz}"]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Locked GPU clock to {freq_mhz} MHz")

        # Verify the lock
        time.sleep(1)  # Give GPU time to adjust
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        actual_freq = int(result.stdout.strip())
        if abs(actual_freq - freq_mhz) > 50:  # Allow some tolerance
            print(f"  Warning: Requested {freq_mhz} MHz but got {actual_freq} MHz")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error locking GPU clock: {e}")
        print(f"  You may need sudo/root permissions: sudo python {sys.argv[0]} ...")
        return False
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return False


def reset_gpu_clock():
    """Reset GPU clock to default using sudo."""
    try:
        subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"], check=True, capture_output=True)
        print("  Reset GPU clock to default")
    except Exception as e:
        print(f"  Error resetting GPU clock: {e}")


def generate_batch_configurations(max_prefill=16, max_decode=64):
    """
    Generate diverse batch configurations for profiling.

    Returns a list of dicts, each with:
        - num_prefill: number of prefill requests
        - num_decode: number of decode requests
        - prefill_lengths: list of prompt lengths for prefill requests
        - decode_kv_lengths: list of KV cache lengths for decode requests
    """
    configs = []

    # Diverse prompt lengths
    prefill_lens = [32, 64, 128, 256, 512, 1024, 2048]
    # Diverse KV cache lengths for decode
    decode_kv_lens = [64, 128, 256, 512, 1024, 2048]

    # 1. Pure prefill batches
    for num_pf in [1, 2, 4, 8, 12, 16]:
        if num_pf > max_prefill:
            continue
        for pf_len in prefill_lens:
            configs.append({
                "num_prefill": num_pf,
                "num_decode": 0,
                "prefill_lengths": [pf_len] * num_pf,
                "decode_kv_lengths": [],
            })

    # 2. Pure decode batches
    for num_dec in [4, 8, 16, 32, 48, 64]:
        if num_dec > max_decode:
            continue
        for kv_len in decode_kv_lens:
            configs.append({
                "num_prefill": 0,
                "num_decode": num_dec,
                "prefill_lengths": [],
                "decode_kv_lengths": [kv_len] * num_dec,
            })

    # 3. Mixed batches (prefill + decode)
    for num_pf in [1, 2, 4]:
        for num_dec in [4, 8, 16, 32]:
            if num_pf > max_prefill or num_dec > max_decode:
                continue
            for pf_len in [128, 512, 1024]:
                for kv_len in [256, 512, 1024]:
                    configs.append({
                        "num_prefill": num_pf,
                        "num_decode": num_dec,
                        "prefill_lengths": [pf_len] * num_pf,
                        "decode_kv_lengths": [kv_len] * num_dec,
                    })

    # 4. Varied lengths within a batch
    configs.append({
        "num_prefill": 4,
        "num_decode": 0,
        "prefill_lengths": [64, 128, 256, 512],
        "decode_kv_lengths": [],
    })

    configs.append({
        "num_prefill": 0,
        "num_decode": 8,
        "prefill_lengths": [],
        "decode_kv_lengths": [128, 256, 512, 1024, 128, 256, 512, 1024],
    })

    print(f"  Generated {len(configs)} batch configurations")
    return configs


def profile_at_frequency(
    freq_mhz,
    model_path,
    batch_configs,
    samples_per_config=3,
    output_file="profiling_data.jsonl",
):
    """
    Profile batches at a specific GPU frequency.

    Args:
        freq_mhz: GPU clock frequency in MHz
        model_path: Path to the LLM model
        batch_configs: List of batch configuration dicts
        samples_per_config: Number of times to repeat each configuration
        output_file: Output JSONL file path
    """
    print(f"\n=== Profiling at {freq_mhz} MHz ===")

    # Lock GPU clock
    if not lock_gpu_clock(freq_mhz):
        print(f"Skipping frequency {freq_mhz} MHz due to clock lock failure")
        return 0

    # Import vLLM after locking clock (to ensure profiling env is set)
    os.environ["VLLM_PROFILING"] = "1"
    os.environ["VLLM_PROFILING_FILE"] = output_file
    os.environ["VLLM_PROFILING_WARMUP"] = "20"

    try:
        from vllm import LLM, SamplingParams

        print(f"  Initializing vLLM with model: {model_path}")
        llm = LLM(
            model=model_path,
            max_model_len=4096,
            max_num_seqs=256,  # Allow large batches
            enforce_eager=True,  # Disable CUDA graphs for profiling
            gpu_memory_utilization=0.9,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # We only need one forward pass
            ignore_eos=True,
        )

        total_samples = 0
        print(f"  Running {len(batch_configs)} batch configs × {samples_per_config} repetitions")

        for config_idx, config in enumerate(batch_configs):
            num_prefill = config["num_prefill"]
            num_decode = config["num_decode"]
            prefill_lengths = config["prefill_lengths"]
            decode_kv_lengths = config["decode_kv_lengths"]

            # Create prompts for this configuration
            prompts = []

            # Prefill requests: create prompts with specified lengths
            for pf_len in prefill_lengths:
                # Generate a simple prompt with approximately pf_len tokens
                # Assume ~4 chars per token
                prompt = "The " + "quick brown fox jumps over the lazy dog. " * (pf_len // 10)
                prompts.append(prompt)

            # Decode requests: we need to simulate ongoing requests with KV cache
            # For simplicity, we'll create long prompts and let the model decode
            # This is a limitation - we can't directly control KV cache length via API
            # A more sophisticated approach would use the LLMEngine directly
            for kv_len in decode_kv_lengths:
                # Create a prompt that will have kv_len tokens already processed
                # This is approximate since we're using the API
                prompt = "The " + "word " * (kv_len // 2)
                prompts.append(prompt)

            # Run this configuration multiple times
            for rep in range(samples_per_config):
                if len(prompts) > 0:
                    try:
                        # Generate - this will trigger the profiling
                        outputs = llm.generate(prompts, sampling_params)
                        total_samples += 1

                        # Print progress every 50 samples
                        if total_samples % 50 == 0:
                            print(f"    Progress: {total_samples} batches profiled at {freq_mhz} MHz")
                    except Exception as e:
                        print(f"    Error during generation: {e}")
                        continue

        print(f"  Completed {total_samples} batch samples at {freq_mhz} MHz")

        # Clean up LLM
        del llm
        torch.cuda.empty_cache()

        return total_samples

    except Exception as e:
        print(f"  Error during profiling at {freq_mhz} MHz: {e}")
        import traceback
        traceback.print_exc()
        return 0

    finally:
        reset_gpu_clock()


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM batch latency across GPU frequencies"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the LLM model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ubuntu/lqs/profiling_data.jsonl",
        help="Output JSONL file path (default: /home/ubuntu/lqs/profiling_data.jsonl)"
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=8,
        help="Number of GPU frequencies to profile (default: 8)"
    )
    parser.add_argument(
        "--samples-per-config",
        type=int,
        default=3,
        help="Number of repetitions per batch configuration (default: 3)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode with fewer configurations"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("vLLM Batch Latency Profiling Script")
    print("=" * 80)

    # Check if running with appropriate permissions
    print("\nChecking GPU clock control permissions...")
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "--lock-gpu-clocks", "210,210"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks"], check=True, capture_output=True)
        print("✓ GPU clock control is available")
    except subprocess.CalledProcessError:
        print("✗ GPU clock control requires root/sudo permissions")
        print("  Please run: sudo python profiling_script.py ...")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error testing GPU clock control: {e}")
        sys.exit(1)

    # Query supported frequencies
    print("\nQuerying supported GPU frequencies...")
    all_freqs = query_supported_gpu_clocks()
    if not all_freqs:
        print("Error: Could not query GPU frequencies")
        sys.exit(1)

    print(f"  Found {len(all_freqs)} supported frequencies")
    print(f"  Range: {min(all_freqs)} - {max(all_freqs)} MHz")

    # Select representative subset
    selected_freqs = select_representative_frequencies(all_freqs, args.num_frequencies)
    print(f"  Selected {len(selected_freqs)} frequencies for profiling:")
    for freq in selected_freqs:
        print(f"    {freq} MHz")

    # Generate batch configurations
    print("\nGenerating batch configurations...")
    if args.quick_test:
        # Quick test mode: fewer configs
        batch_configs = generate_batch_configurations(max_prefill=8, max_decode=32)
        batch_configs = batch_configs[:50]  # Limit to 50 configs
        print(f"  Quick test mode: using {len(batch_configs)} configurations")
    else:
        batch_configs = generate_batch_configurations(max_prefill=16, max_decode=64)

    # Clear output file if it exists
    output_path = Path(args.output)
    if output_path.exists():
        print(f"\nRemoving existing profiling data: {args.output}")
        output_path.unlink()

    # Profile at each frequency
    print("\nStarting profiling...")
    print(f"  Output file: {args.output}")
    print(f"  Samples per config: {args.samples_per_config}")

    total_samples = 0
    for freq_idx, freq_mhz in enumerate(selected_freqs):
        print(f"\nFrequency {freq_idx + 1}/{len(selected_freqs)}: {freq_mhz} MHz")

        samples = profile_at_frequency(
            freq_mhz=freq_mhz,
            model_path=args.model_path,
            batch_configs=batch_configs,
            samples_per_config=args.samples_per_config,
            output_file=args.output,
        )

        total_samples += samples

    # Reset GPU clock at the end
    print("\n" + "=" * 80)
    print(f"Profiling complete!")
    print(f"  Total samples collected: {total_samples}")
    print(f"  Output file: {args.output}")
    print("=" * 80)

    reset_gpu_clock()


if __name__ == "__main__":
    main()
