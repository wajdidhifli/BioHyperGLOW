#!/usr/bin/env python3
"""Download external benchmark datasets and ontology for BioHyperGLOW.

This script fetches the official MONDO Disease Ontology release and the
DisGeNET benchmark association datasets used in the BioHyperGLOW paper.

Sources:
1. MONDO Disease Ontology:
   - Monarch Initiative release (CC BY 4.0)
   - URL: https://github.com/monarch-initiative/mondo/releases/download/v2026-11-03/mondo.obo
   - SHA-256: b5a3d848578694601dffb1c2866798d4f6f4b4a12c315030c4d4c62ffe4e7e22

2. DisGeNET-v1 and v2 Benchmark Associations:
   - Zenodo benchmark repository: https://doi.org/10.5281/zenodo.21886712
   - DisGeNET-v1: FactorHNE benchmark source (130,820 positive associations)
     URL: https://raw.githubusercontent.com/xasdzxc/FactorHNE/master/data/DisGeNet_dis_gene.txt
   - DisGeNET-v2: HOGCN-LP GDI benchmark source (81,746 positive associations)
     URL: https://raw.githubusercontent.com/kckishan/HOGCN-LP/master/data/GDI/fold1/{train,val,test}.csv

Usage:
    # Download MONDO ontology and DisGeNET-v1 benchmark:
    python examples/download_data.py --dataset v1

    # Download both v1 and v2 benchmarks:
    python examples/download_data.py --dataset all

    # Download only the ontology:
    python examples/download_data.py --dataset none
"""

import argparse
import csv
import hashlib
import io
import os
import sys
import urllib.request
from pathlib import Path


MONDO_RELEASES = {
    "v2026-11-03": {
        "url": "https://github.com/monarch-initiative/mondo/releases/download/v2026-11-03/mondo.obo",
        "sha256": "b5a3d848578694601dffb1c2866798d4f6f4b4a12c315030c4d4c62ffe4e7e22",
        "size_mb": 51.0,
    }
}

FACTORHNE_V1_URL = (
    "https://raw.githubusercontent.com/xasdzxc/FactorHNE/master/data/DisGeNet_dis_gene.txt"
)
HOGCN_V2_URLS = [
    "https://raw.githubusercontent.com/kckishan/HOGCN-LP/master/data/GDI/fold1/train.csv",
    "https://raw.githubusercontent.com/kckishan/HOGCN-LP/master/data/GDI/fold1/val.csv",
    "https://raw.githubusercontent.com/kckishan/HOGCN-LP/master/data/GDI/fold1/test.csv",
]


def download_stream(url: str, dest_path: Path, expected_sha256: str = None) -> Path:
    """Download a file with progress reporting and optional SHA-256 verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Destination:     {dest_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "BioHyperGLOW-Downloader/0.1.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 65536
        downloaded = 0
        sha = hashlib.sha256()

        temp_path = dest_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                sha.update(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_dl = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {mb_dl:5.1f} / {mb_tot:5.1f} MB ({percent:4.1f}%)")
                else:
                    mb_dl = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r  Downloaded: {mb_dl:5.1f} MB")
                sys.stdout.flush()

        print()

    digest = sha.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 integrity mismatch for {dest_path.name}!\n"
            f"  Expected: {expected_sha256}\n"
            f"  Computed: {digest}"
        )
    temp_path.replace(dest_path)
    print(f"Saved: {dest_path} (SHA-256: {digest[:12]}...)")
    return dest_path


def fetch_disgenet_v1(out_dir: Path) -> Path:
    """Fetch DisGeNET-v1 benchmark associations and write clean CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_csv = out_dir / "disgenet_v1_associations.csv"
    if target_csv.exists():
        print(f"DisGeNET-v1 association file already exists: {target_csv}")
        return target_csv

    print("\n--- Fetching DisGeNET-v1 Benchmark (130,820 associations) ---")
    req = urllib.request.Request(
        FACTORHNE_V1_URL, headers={"User-Agent": "BioHyperGLOW-Downloader/0.1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")

    pairs = set()
    for line in content.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            pairs.add((parts[0].strip(), parts[1].strip()))

    sorted_pairs = sorted(pairs)
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "gene"])
        writer.writerows(sorted_pairs)

    print(f"DisGeNET-v1 saved: {target_csv} ({len(sorted_pairs):,} associations)")
    return target_csv


def fetch_disgenet_v2(out_dir: Path) -> Path:
    """Fetch DisGeNET-v2 benchmark associations and write clean CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_csv = out_dir / "disgenet_v2_associations.csv"
    if target_csv.exists():
        print(f"DisGeNET-v2 association file already exists: {target_csv}")
        return target_csv

    print("\n--- Fetching DisGeNET-v2 Benchmark (81,746 associations) ---")
    pairs = set()
    for url in HOGCN_V2_URLS:
        print(f"Fetching partition: {url.split('/')[-1]}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "BioHyperGLOW-Downloader/0.1.0"}
        )
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")

        reader = csv.reader(io.StringIO(content))
        header = next(reader, None)
        for row in reader:
            if len(row) >= 4:
                # row columns: Unnamed: 0, Gene_ID, Disease_ID, label
                gene, disease, label = row[1].strip(), row[2].strip(), row[3].strip()
                if label == "1":
                    pairs.add((disease, gene))

    sorted_pairs = sorted(pairs)
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "gene"])
        writer.writerows(sorted_pairs)

    print(f"DisGeNET-v2 saved: {target_csv} ({len(sorted_pairs):,} associations)")
    return target_csv


def fetch_mondo_ontology(onto_dir: Path, release: str = "v2026-11-03") -> Path:
    """Download MONDO disease ontology OBO release."""
    onto_dir.mkdir(parents=True, exist_ok=True)
    dest_path = onto_dir / "mondo.obo"
    if dest_path.exists():
        print(f"MONDO ontology file already exists: {dest_path}")
        return dest_path

    if release not in MONDO_RELEASES:
        raise ValueError(
            f"Unknown MONDO release: {release}. Available: {list(MONDO_RELEASES.keys())}"
        )

    info = MONDO_RELEASES[release]
    print(f"\n--- Downloading MONDO Disease Ontology ({release}, ~{info['size_mb']} MB) ---")
    return download_stream(info["url"], dest_path, expected_sha256=info["sha256"])


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark associations and ontology for BioHyperGLOW."
    )
    parser.add_argument(
        "--dataset",
        choices=["v1", "v2", "all", "none"],
        default="v1",
        help="Which DisGeNET benchmark to download (default: v1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory where association CSVs will be stored (default: 'data').",
    )
    parser.add_argument(
        "--onto-dir",
        type=Path,
        default=Path("ontology"),
        help="Directory where ontology OBO will be stored (default: 'ontology').",
    )
    parser.add_argument(
        "--skip-ontology",
        action="store_true",
        help="Skip downloading the MONDO ontology.",
    )
    parser.add_argument(
        "--onto-version",
        default="v2026-11-03",
        help="MONDO release version to download (default: 'v2026-11-03').",
    )

    args = parser.parse_args()

    print("============================================================")
    print("BioHyperGLOW External Benchmark & Ontology Downloader")
    print("============================================================")

    # 1. Ontology
    if not args.skip_ontology:
        fetch_mondo_ontology(args.onto_dir, release=args.onto_version)

    # 2. Benchmark datasets
    if args.dataset in ("v1", "all"):
        fetch_disgenet_v1(args.out_dir)
    if args.dataset in ("v2", "all"):
        fetch_disgenet_v2(args.out_dir)

    print("\n============================================================")
    print("Download and setup complete!")
    print("You can now train BioHyperGLOW using:")
    print("  biohyperglow train \\")
    print(f"      --edges {args.out_dir}/disgenet_{'v1' if args.dataset != 'v2' else 'v2'}_associations.csv \\")
    print(f"      --ontology {args.onto_dir}/mondo.obo \\")
    print("      --out checkpoints/biohyperglow_model.npz \\")
    print("      --preset paper")
    print("============================================================")


if __name__ == "__main__":
    main()
