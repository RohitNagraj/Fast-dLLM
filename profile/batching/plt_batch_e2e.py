import re
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Raw input text (paste yours here)
# -----------------------------
SEQ_LEN=1348
BLK_SIZE=8

raw = """
[ 1] (dual cache): 10.695560693740845
[ 1] ( original ): 37.914669036865234
[ 2] (dual cache): 16.381134510040283
[ 2] ( original ): 70.85624861717224
[ 4] (dual cache): 28.541807174682617
[ 4] ( original ): 136.8460955619812
"""

# -----------------------------
# Parsing
# -----------------------------
pattern = r"\[\s*(\d+)\s*\]\s*\(([^)]+)\):\s*([0-9.]+)"

dual_times = {}
orig_times = {}

for batch, method, value in re.findall(pattern, raw):
    print("Parsed:", batch, method, value)
    batch = int(batch)
    value = float(value)
    method = method.strip().lower()
    if "dual" in method:
        dual_times[batch] = value
    else:
        orig_times[batch] = value

batches = sorted(dual_times.keys())

# -----------------------------
# Throughput calculation
# throughput = 128 * batch_size / compute_time
# -----------------------------
dual_tp = [128 * b / dual_times[b] for b in batches]
orig_tp = [128 * b / orig_times[b] for b in batches]

for b in batches:
    print(b, dual_times[b], orig_times[b])
print("Dual Throughput:", dual_tp)
print("Original Throughput:", orig_tp)

# Ideal linear scaling: scaled from the throughput at batch=1
ideal_start = dual_tp[0]
ideal = [ideal_start * b for b in batches]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(batches, dual_tp, "-o", label="Dual Cache", color="#ECB81C")
plt.plot(batches, orig_tp, "-s", label="Original", color="#1EA1F8")
plt.plot(batches, ideal, "r--", label="Ideal Linear Scaling")

plt.title(f"Throughput vs Batch Size\n(with block_size={BLK_SIZE})", fontsize=18)
plt.xlabel("Batch Size", fontsize=14)
plt.ylabel("Throughput (tokens/s)", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f"batch_e2e_seq{SEQ_LEN}_blk{BLK_SIZE}.png")
