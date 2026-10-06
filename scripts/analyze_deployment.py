# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze a deployment recording without creating DDS endpoints."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Directory containing metadata.json and recording.sqlite3.")
    parser.add_argument("--out", help="Report directory; defaults to RECORDING/analysis.")
    parser.add_argument("--no_plots", action="store_true", help="Export deploynose data without static PNG plots.")
    args = parser.parse_args()
    from g1_deploy.bootstrap import ensure_cyclonedds
    ensure_cyclonedds()
    from g1_deploy.analysis import save_report
    output, summary = save_report(args.recording, args.out, plots=not args.no_plots)
    print(f"[report] {output}")
    from pathlib import Path
    print(f"[deploynose] {output / (Path(args.recording).resolve().name + '.deploynose')}")
    print(f"[data] {summary['samples']} state samples, {summary['recorded_seconds']:.1f}s")
    print(f"[integrity] clean shutdown={summary['clean_shutdown']}, dropped={summary['dropped']}, "
          f"writer error={summary['writer_error']}")
    print("[scope] No measured base linear velocity. "
          "Spectral peaks are diagnostic indicators, not fault classifications.")


if __name__ == "__main__":
    main()
