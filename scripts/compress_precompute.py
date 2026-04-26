#!/usr/bin/env python3
"""In-place convert all precompute files from .json.gz → .mpk.zst.

~5x size reduction vs gzip while preserving decision-level strategy
correctness (u8 quantization gives 1/255 precision, below CFR+ convergence
error). Processes all action_line dirs under precompute_out/ in parallel.

Safe semantics: write .mpk.zst.tmp → rename → delete .json.gz only after
successful rename. Safe to interrupt / resume (skips already-converted files).

Usage:
    python3 scripts/compress_precompute.py [--dry-run] [--workers N]
                                           [--dir <path>]
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skill.poker.lib.compressed_format import convert_file  # noqa: E402


def find_convertable(root: Path):
    """Yield (src_gz_path, dst_mpk_zst_path) for every .json.gz that has no
    sibling .mpk.zst with the same flop stem."""
    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        for gz in sorted(job_dir.glob("flop_*.json.gz")):
            stem = gz.name[:-len(".json.gz")]
            mpk = job_dir / f"{stem}.mpk.zst"
            if mpk.exists():
                continue  # already converted
            yield gz, mpk


def do_one(pair):
    gz, mpk = pair
    tmp = mpk.with_suffix(mpk.suffix + ".tmp")
    before = gz.stat().st_size
    try:
        after = convert_file(gz, tmp)
        tmp.rename(mpk)
        gz.unlink()
        return (before, after, None)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return (before, 0, str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "precompute_out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = list(find_convertable(args.dir))
    if not pairs:
        print("Nothing to convert.")
        return
    print(f"Found {len(pairs)} .json.gz files to convert.")
    if args.dry_run:
        for gz, _ in pairs[:10]:
            print(f"  {gz.relative_to(args.dir)}")
        print("  ... (dry run; no changes made)")
        return

    t0 = time.time()
    total_before = 0; total_after = 0; errors = 0
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for before, after, err in ex.map(do_one, pairs, chunksize=4):
            total_before += before
            total_after += after
            done += 1
            if err:
                errors += 1
                print(f"error: {err}", file=sys.stderr)
            if done % 200 == 0:
                ratio = total_before / max(total_after, 1)
                print(f"  {done}/{len(pairs)}  "
                      f"{total_before / 1024**2:.0f} → {total_after / 1024**2:.0f} MB "
                      f"({ratio:.1f}x)  elapsed {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    ratio = total_before / max(total_after, 1)
    print(f"\nDone. Converted {done} files in {elapsed:.0f}s.")
    print(f"Before: {total_before / 1024**3:.2f} GB")
    print(f"After:  {total_after / 1024**3:.2f} GB")
    print(f"Ratio:  {ratio:.1f}x smaller")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
