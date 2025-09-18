from pathlib import Path
import sys
import numpy as np

################################################################################
# This script calculates the average duration of the audio signals in the dataset
################################################################################

SR = 4000

# scripts/ is under repo root; go up one to reach root and then data/signals
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "signals"

def main() -> int:
    files = sorted(DATA_ROOT.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {DATA_ROOT}", file=sys.stderr)
        return 1
    durations = []
    for p in files:
        x = np.load(str(p), allow_pickle=False)
        n = x.shape[0]
        durations.append(n / SR)
    avg = float(np.mean(durations))
    print(f"avg_duration_s: {avg:.2f} ({len(files)} files)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


