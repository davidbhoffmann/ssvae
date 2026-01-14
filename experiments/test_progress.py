#!/usr/bin/env python3
"""
Simple test to demonstrate progress bars are working
"""

from tqdm import tqdm
import time

print("\n" + "=" * 80)
print("Testing Progress Bars")
print("=" * 80 + "\n")

# Test 1: Simple progress bar
print("Test 1: Basic progress bar")
for i in tqdm(range(10), desc="Basic Test"):
    time.sleep(0.1)

print()

# Test 2: Nested progress bars
print("Test 2: Nested progress bars (like experiments)")
experiments = [(0.1, 0.0), (0.1, 0.1), (0.05, 0.0), (0.05, 0.1)]

for label_frac, noise in tqdm(experiments, desc="Experiments", position=0):
    desc = f"Labels:{label_frac*100:.0f}% Noise:{noise*100:.0f}%"

    # Simulate epochs
    pbar = tqdm(range(5), desc=desc, leave=False)
    for epoch in pbar:
        time.sleep(0.1)
        pbar.set_postfix(
            {"Acc": f"{0.7 + epoch*0.05:.3f}", "ELBO": f"{-0.9 + epoch*0.01:.3e}"}
        )

print("\n" + "=" * 80)
print("✓ Progress bars working correctly!")
print("=" * 80)
