"""Generate the synthetic benchmark panel without downloading any data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from quickstart import make_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=int, default=1000)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, default=Path(".local/benchmarks/input_panel.dta"))
    args = parser.parse_args()
    if args.firms < 2 or args.years < 1:
        parser.error("Use at least two firms and one year")
    data = make_panel(args.firms, args.years, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_stata(args.output, write_index=False, version=118)
    print(f"Wrote {len(data):,} synthetic firm-year observations to {args.output}")


if __name__ == "__main__":
    main()
