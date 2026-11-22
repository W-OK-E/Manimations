"""Render an approved slug or import an existing clip, then export MP4/poster/indexes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from catalog import ROOT, main as publish_catalog, read_source, write_json


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("--input", type=Path, help="Export an existing approved clip; do not render a scene")
    parser.add_argument("--scene-class", help="Required when rendering a reviewed scene rather than importing")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs",
                        help="Export a separate review tree without overwriting working docs/")
    args = parser.parse_args()
    entry = next((item for item in read_source() if item["slug"] == args.slug), None)
    if entry is None:
        parser.error("Add the slug's title, tags and description to animations/catalog.json first")
    source = args.input
    if source is None:
        scene = ROOT / "animations" / args.slug / "scene.py"
        design = scene.with_name("DESIGN.md")
        if not scene.is_file() or not design.is_file() or not args.scene_class:
            parser.error("Rendering requires scene.py, reviewed DESIGN.md and --scene-class; otherwise use --input")
        output = ROOT / "build" / args.slug
        run(sys.executable, "-m", "manim", "-qm", "--renderer", "cairo", "--media_dir", output,
            "--format", "mp4", scene, args.scene_class)
        source = output / "videos" / "scene" / "720p30" / f"{args.scene_class}.mp4"
    source = source.resolve()
    details = probe(source)
    video_stream = next(stream for stream in details["streams"] if stream["codec_type"] == "video")
    duration = float(details["format"]["duration"])
    media = args.output_dir.resolve() / "media"
    media.mkdir(parents=True, exist_ok=True)
    video = media / f"{args.slug}.mp4"
    poster = media / f"{args.slug}.webp"
    # Bound resolution without upscaling; deliver 30 fps and fast-start H.264.
    run("ffmpeg", "-v", "error", "-y", "-i", source, "-map", "0:v:0", "-map", "0:a?",
        "-vf", "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30",
        "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", video)
    poster_time = min(1.5, duration / 2)
    for quality in (65, 50, 35, 20):
        run("ffmpeg", "-v", "error", "-y", "-ss", str(poster_time), "-i", video,
            "-frames:v", "1", "-vf", "scale=640:-2", "-c:v", "libwebp", "-quality", str(quality), poster)
        if poster.stat().st_size <= 15_000:
            break
    result = dict(entry, duration=duration, video=f"media/{video.name}", poster=f"media/{poster.name}")
    write_json(media / f"{args.slug}.json", result)
    publish_catalog(args.output_dir.resolve())
    print(f"Video: {video.stat().st_size:,} bytes; poster: {poster.stat().st_size:,} bytes")
    print(f"Source: {video_stream['width']}×{video_stream['height']}; duration: {duration:g}s")
    if poster.stat().st_size > 15_000:
        print("Poster exceeds 15 KB target; review before publication.")
    print("Outputs remain local; committing binaries and publishing require om's approval.")


if __name__ == "__main__":
    main()
