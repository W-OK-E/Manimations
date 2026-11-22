"""Assemble a GitHub Pages artifact from explicitly selected local/release assets."""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from catalog import ROOT, publish, read_source


def plan(config_path):
    config = json.loads(Path(config_path).read_text())
    tag = config.get("release_tag", "")
    slugs = config.get("approved_slugs", [])
    if not tag or not slugs:
        raise ValueError("TODO(om): fill release_tag and approved_slugs in integration/pages.json before publishing")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("Use a simple release tag containing only letters, numbers, dots, underscores or hyphens")
    if not isinstance(slugs, list) or not all(isinstance(s, str) for s in slugs):
        raise ValueError("approved_slugs must be a list of catalog slugs")
    known = {item["slug"] for item in read_source()}
    if len(slugs) != len(set(slugs)) or any(s not in known for s in slugs):
        raise ValueError("approved_slugs must contain unique slugs from animations/catalog.json")
    if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", s) for s in slugs):
        raise ValueError("Invalid catalog slug")
    return {"release_tag": tag, "slugs": slugs,
            "assets": [f"{slug}.{extension}" for slug in slugs for extension in ("mp4", "webp", "json")]}


def stage(config_path, media_dir, destination):
    selected = plan(config_path)
    media_dir, destination = Path(media_dir), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a new/empty artifact directory so old, unselected media cannot be published")
    source = {item["slug"]: item for item in read_source()}
    entries, files = [], []
    for slug in selected["slugs"]:
        assets = [media_dir / f"{slug}.{extension}" for extension in ("mp4", "webp", "json")]
        if any(not file.is_file() or file.is_symlink() for file in assets):
            raise ValueError(f"Missing regular MP4/WebP/JSON assets for {slug}; export/download them first")
        metadata = json.loads(assets[2].read_text())
        if metadata.get("slug") != slug or metadata.get("video") != f"media/{slug}.mp4" or metadata.get("poster") != f"media/{slug}.webp":
            raise ValueError(f"Media metadata paths do not match {slug}")
        duration = metadata.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration for {slug}")
        metadata.update(source[slug])
        entries.append(metadata)
        files.extend(assets)
    frontend = [ROOT / "docs" / name for name in ("index.html", "gallery.css", "gallery.js")]
    total = sum(file.stat().st_size for file in files + frontend)
    if total >= 999_000_000:
        raise ValueError("Artifact exceeds the staging budget below GitHub Pages' 1 GB site limit")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "media").mkdir(exist_ok=True)
    for file in frontend:
        shutil.copy2(file, destination / file.name)
    for file in files:
        shutil.copy2(file, destination / "media" / file.name)
    publish(entries, destination)
    (destination / ".nojekyll").touch()
    report = {"directory": str(destination), "release_tag": selected["release_tag"],
              "slugs": selected["slugs"], "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "integration/pages.json")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "build/release-media")
    parser.add_argument("--destination", type=Path, default=ROOT / "build/pages")
    parser.add_argument("--write-plan", type=Path, help="Validate choices and write the exact GitHub Release download plan only")
    args = parser.parse_args()
    try:
        if args.write_plan:
            value = plan(args.config)
            args.write_plan.parent.mkdir(parents=True, exist_ok=True)
            args.write_plan.write_text(json.dumps(value, indent=2) + "\n")
            print(f"Validated {len(value['slugs'])} selected clips; no download or deployment performed.")
        else:
            stage(args.config, args.media_dir, args.destination)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
