"""``pert report`` - write the plain-language report of a finished run.
"""

from __future__ import annotations

import argparse
import os

from pert.reporting.reader_report import write_reader_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None, help="Output Markdown path (default: <run-dir>/analysis/reader_report.md)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(run_dir, "analysis", "reader_report.md")
    out_path = os.path.abspath(out_path)

    written = write_reader_report(run_dir, out_path)
    print(written)


if __name__ == "__main__":
    main()
