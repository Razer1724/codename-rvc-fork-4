import argparse
import sys
import numpy as np


def parse_measurements(filepath: str) -> list[float]:
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                try:
                    v = float(line.split('=')[-1])
                    if v > 1e-6:  # skip mute/silent zeros
                        values.append(v)
                except ValueError:
                    continue
            else:
                try:
                    v = float(line)
                    if v > 1e-6:
                        values.append(v)
                except ValueError:
                    continue
    return values


def compute_stats(values: list[float], settled_n: int) -> dict:
    all_arr = np.array(values)
    settled_arr = np.array(values[-settled_n:])

    # remove outliers beyond 1.5 IQR before computing settled stats
    q1, q3 = np.percentile(settled_arr, [25, 75])
    iqr = q3 - q1
    mask = (settled_arr >= q1 - 1.5 * iqr) & (settled_arr <= q3 + 1.5 * iqr)
    settled_clean = settled_arr[mask]

    mid = len(settled_clean) // 2
    first_half = float(np.mean(settled_clean[:mid]))
    second_half = float(np.mean(settled_clean[mid:]))
    decline_pct = (first_half - second_half) / (first_half + 1e-8) * 100
    cv = float(np.std(settled_clean) / (np.mean(settled_clean) + 1e-8))

    return {
        "total_steps":    len(values),
        "settled_n":      settled_n,
        "settled_clean_n":int(mask.sum()),
        "settled_mean":   float(np.mean(settled_clean)),
        "settled_median": float(np.median(settled_clean)),
        "settled_std":    float(np.std(settled_clean)),
        "settled_min":    float(np.min(settled_clean)),
        "settled_max":    float(np.max(settled_clean)),
        "overall_mean":   float(np.mean(all_arr)),
        "first_value":    float(values[0]),
        "last_value":     float(values[-1]),
        "cv":             cv,
        "decline_pct":    decline_pct,
    }


def stability_verdict(stats: dict) -> tuple[str, str]:
    cv = stats["cv"]
    dp = stats["decline_pct"]
    if cv < 0.05 and abs(dp) < 1.0:
        return "STABLE", "Good to use — loss has settled."
    elif cv < 0.10 and abs(dp) < 3.0:
        return "PROBABLY STABLE", "Likely fine — run 20-30 more steps to confirm."
    elif dp > 3.0:
        return "STILL DECLINING", f"Loss dropped {dp:.1f}% across settled window — run more steps."
    else:
        return "NOISY", f"High variance (CV={cv:.3f}) — check for spikes or run more steps."


def print_file_stats(label: str, stats: dict):
    verdict, advice = stability_verdict(stats)
    print(f"\n  [{label}]")
    print(f"  Total steps:       {stats['total_steps']}")
    print(f"  First → Last:      {stats['first_value']:.4f} → {stats['last_value']:.4f}")
    print(f"  Overall mean:      {stats['overall_mean']:.4f}")
    print(f"  Settled mean:      {stats['settled_mean']:.4f}")
    print(f"  Settled median:    {stats['settled_median']:.4f}")
    print(f"  Settled std:       {stats['settled_std']:.4f}  (CV={stats['cv']:.3f})")
    print(f"  Settled range:     {stats['settled_min']:.4f} – {stats['settled_max']:.4f}")
    print(f"  Decline in window: {stats['decline_pct']:+.2f}%")
    print(f"  Stability:         {verdict}")
    print(f"  → {advice}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate loss weight from raw measurement log.")
    parser.add_argument("filepath", help="Path to .txt file with raw loss measurements to calibrate")
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference loss .txt file (e.g. raw_mel.txt) for ratio calibration")
    parser.add_argument("--ref-weight", type=float, default=45.0,
                        help="Known weight of the reference loss (default: 45)")
    parser.add_argument("--target", type=float, default=45.0,
                        help="Target effective contribution when no --ref provided (default: 45)")
    parser.add_argument("--settled", type=int, default=30,
                        help="Number of final steps to use as settled window (default: 30)")
    args = parser.parse_args()

    values = parse_measurements(args.filepath)
    if not values:
        print("ERROR: No valid measurements found in primary file.")
        sys.exit(1)
    settled = min(args.settled, len(values))
    if len(values) < args.settled:
        print(f"WARNING: Only {len(values)} steps in primary file. Using all as settled window.")
    stats = compute_stats(values, settled)

    print()
    print("=" * 58)
    print("  LOSS WEIGHT CALIBRATION REPORT")
    print("=" * 58)
    print(f"  Primary file:    {args.filepath}")
    print(f"  Settled window:  last {settled} steps")
    print("-" * 58)

    if args.ref is not None:
        ref_values = parse_measurements(args.ref)
        if not ref_values:
            print("ERROR: No valid measurements found in reference file.")
            sys.exit(1)
        ref_settled = min(args.settled, len(ref_values))
        if len(ref_values) < args.settled:
            print(f"WARNING: Only {len(ref_values)} steps in reference file. Using all as settled window.")
        ref_stats = compute_stats(ref_values, ref_settled)

        weight_mean   = args.ref_weight * (ref_stats["settled_mean"]   / stats["settled_mean"])
        weight_median = args.ref_weight * (ref_stats["settled_median"] / stats["settled_median"])

        print(f"  Reference file:  {args.ref}")
        print(f"  Reference weight:{args.ref_weight}")
        print(f"  Mode:            ratio calibration")

        print_file_stats("REFERENCE LOSS", ref_stats)
        print_file_stats("NEW LOSS (to calibrate)", stats)

        print()
        print("=" * 58)
        print("  RESULT")
        print("=" * 58)
        ratio = ref_stats["settled_mean"] / stats["settled_mean"]
        print(f"  Ratio (ref/new): {ref_stats['settled_mean']:.4f} / {stats['settled_mean']:.4f} = {ratio:.4f}")
        print(f"  Weight (mean):   {weight_mean:.2f}  →  rounded: {round(weight_mean)}")
        print(f"  Weight (median): {weight_median:.2f}  →  rounded: {round(weight_median)}")
        print()
        print("  Recommended:")
        print(f"    loss = fn_new_loss(y_hat, y) * {round(weight_mean)}")
        print(f"    # matches fn_ref_loss * {args.ref_weight} in effective contribution")

    else:
        weight_mean   = args.target / stats["settled_mean"]
        weight_median = args.target / stats["settled_median"]

        print(f"  Target contrib:  {args.target}")
        print(f"  Mode:            single loss calibration")

        print_file_stats("LOSS", stats)

        print()
        print("=" * 58)
        print("  RESULT")
        print("=" * 58)
        print(f"  Weight (mean):   {weight_mean:.2f}  →  rounded: {round(weight_mean)}")
        print(f"  Weight (median): {weight_median:.2f}  →  rounded: {round(weight_median)}")
        print()
        print("  Recommended:")
        print(f"    loss = fn_loss(y_hat, y) * {round(weight_mean)}")

    print()


if __name__ == "__main__":
    main()
