"""Build a 10,000-MID selection with complete stacks, starting from the 6,000 sample.

Inputs (in the folder of this script):
    A00_G07_MAD_6000_sample_MID.txt   MIDs already downloaded (one per line)
    output/output.csv                 all MIDs, with columns StackID, SST, MasterindexID

Steps:
    1. Every stack touched by the sample is completed with all its MIDs.
    2. The remaining class deficits (default A00=4000, G07=3000, MAD=3000) are
       filled with whole, randomly drawn stacks that are not in the selection yet.
       A stack is taken only when it helps a class still below target more than
       it overshoots classes already full.
    3. Writes (no header, one MID per line):
         output/selected_mids.txt      the full selection (sample included)
         output/to_download_mids.txt   only the MIDs not in the sample -> AWS export

Counts are MIDs per class. Complete stacks make exact targets impossible; the
final counts are printed.

    python select_10000_mids.py
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CLASSES = ("A00", "G07", "MAD")
COLUMN_ALIASES = {"stackid": "stack_id", "stack_id": "stack_id", "sst": "sst",
                  "masterindexid": "masterindex_id", "masterindex_id": "masterindex_id", "mid": "masterindex_id"}
SEPARATORS = [",", ";", "\t", "|"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sample-file", default="A00_G07_MAD_6000_sample_MID.txt")
    parser.add_argument("--all-file", default="output/output.csv")
    parser.add_argument("--target-a00", type=int, default=4000)
    parser.add_argument("--target-g07", type=int, default=3000)
    parser.add_argument("--target-mad", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def read_sample(path: Path) -> set[str]:
    """MIDs of the existing sample; a header line, if any, is ignored."""
    mids = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    return {m for m in mids if m.lower() not in {"masterindex_id", "masterindexid", "mid"}}


def read_all(path: Path) -> pd.DataFrame:
    """All MIDs with stack_id and sst; the separator is detected from the header."""
    for separator in SEPARATORS:
        data = pd.read_csv(path, sep=separator, dtype=str, encoding="utf-8-sig")
        data.columns = [str(c).strip().lower() for c in data.columns]
        data = data.rename(columns=COLUMN_ALIASES)
        if {"stack_id", "sst", "masterindex_id"} <= set(data.columns):
            break
    else:
        raise ValueError(f"{path} needs the columns stack_id, sst and masterindex_id (found: {list(data.columns)}).")
    data = data[["stack_id", "sst", "masterindex_id"]].dropna()
    data = data.apply(lambda column: column.str.strip())
    data["sst"] = data["sst"].str.upper()
    return data.drop_duplicates()


def class_counts(mids: set[str], mid_classes: dict[str, set[str]]) -> Counter:
    """MIDs per class (a MID with several classes counts in each)."""
    counts = Counter()
    for mid in mids:
        counts.update(mid_classes.get(mid, set()))
    return counts


def main() -> None:
    args = build_parser().parse_args()
    targets = {"A00": args.target_a00, "G07": args.target_g07, "MAD": args.target_mad}
    sample = read_sample(args.root / args.sample_file)
    data = read_all(args.root / args.all_file)

    mid_classes: dict[str, set[str]] = defaultdict(set)
    for mid, sst in zip(data["masterindex_id"], data["sst"]):
        if sst in CLASSES:
            mid_classes[mid].add(sst)
    stack_mids = data.groupby("stack_id")["masterindex_id"].agg(set).to_dict()
    mid_stacks = data.groupby("masterindex_id")["stack_id"].agg(set).to_dict()

    unknown = sample - set(mid_stacks)
    if unknown:
        print(f"WARNING: {len(unknown)} sample MIDs are not in {args.all_file}; they are kept but cannot be completed.")
    multi_class = sum(len(c) > 1 for c in mid_classes.values())
    if multi_class:
        print(f"NOTE: {multi_class} MIDs carry more than one class; they count in each.")

    # 1. complete the stacks of the sample
    selected_stacks = {s for mid in sample for s in mid_stacks.get(mid, set())}
    selected = set(sample) | {mid for s in selected_stacks for mid in stack_mids[s]}
    print(f"Sample: {len(sample):,} MIDs, counts {dict(class_counts(sample, mid_classes))}")
    print(f"After completing {len(selected_stacks):,} stacks: {len(selected):,} MIDs, "
          f"counts {dict(class_counts(selected, mid_classes))}")

    # 2. fill the deficits with whole random stacks
    candidates = sorted(set(stack_mids) - selected_stacks)
    random.Random(args.seed).shuffle(candidates)
    counts = class_counts(selected, mid_classes)
    for stack in candidates:
        deficit = {c: targets[c] - counts[c] for c in CLASSES}
        if all(d <= 0 for d in deficit.values()):
            break
        new_mids = stack_mids[stack] - selected
        if not new_mids:
            continue
        added = class_counts(new_mids, mid_classes)
        gain = sum(min(added[c], max(deficit[c], 0)) for c in CLASSES)
        overshoot = sum(added[c] - max(deficit[c], 0) for c in CLASSES if added[c] > max(deficit[c], 0))
        if gain > 0 and gain >= overshoot:
            selected |= new_mids
            selected_stacks.add(stack)
            counts += added

    # 3. write outputs
    to_download = selected - sample
    output_dir = args.root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_mids.txt").write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    (output_dir / "to_download_mids.txt").write_text("\n".join(sorted(to_download)) + "\n", encoding="utf-8")

    print(f"\nFinal: {len(selected):,} MIDs in {len(selected_stacks):,} complete stacks")
    for c in CLASSES:
        print(f"  {c}: {counts[c]:,} (target {targets[c]:,})")
    print(f"To download: {len(to_download):,} MIDs -> {output_dir / 'to_download_mids.txt'}")
    print(f"Full selection: {output_dir / 'selected_mids.txt'}")


if __name__ == "__main__":
    main()
