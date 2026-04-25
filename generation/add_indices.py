"""Add index field to existing JSONL files so they can be matched across variants.

Reads each JSONL file, adds {"index": i} to every record (i = line position),
and writes back in-place. Skips files that already have indices.

Usage:
    # Add indices to all files in a directory:
    python -m generation.add_indices data/generated/final_all/gemini-2.5-pro/

    # Add indices to a single file:
    python -m generation.add_indices data/generated/final_all/gemini-2.5-pro/medical_advice.jsonl

    # Dry run (show what would change):
    python -m generation.add_indices data/generated/final_all/gemini-2.5-pro/ --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def add_indices_to_file(path: Path, dry_run: bool = False) -> tuple[int, bool]:
    """Add index field to every record in a JSONL file.

    Returns (record_count, was_modified).
    """
    records = []
    already_indexed = True

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if "index" not in rec:
                    already_indexed = False
                records.append(rec)

    if already_indexed:
        return len(records), False

    if dry_run:
        return len(records), True

    # Add indices and write back
    for i, rec in enumerate(records):
        rec["index"] = i

    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return len(records), True


def main():
    parser = argparse.ArgumentParser(description="Add index field to JSONL files")
    parser.add_argument(
        "path", type=str,
        help="Path to a JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would change without modifying files"
    )

    args = parser.parse_args()
    target = Path(args.path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.jsonl"))
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    if not files:
        print(f"No JSONL files found in {target}")
        return

    action = "Would modify" if args.dry_run else "Modified"
    for fpath in files:
        count, modified = add_indices_to_file(fpath, dry_run=args.dry_run)
        if modified:
            print(f"  {action}: {fpath.name} ({count} records)")
        else:
            print(f"  Already indexed: {fpath.name} ({count} records)")

    if args.dry_run:
        print("\n  (dry run — no files were changed)")


if __name__ == "__main__":
    main()
