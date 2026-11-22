"""Decode the completed preview and extract one labelled inspection frame per shot."""

import json
import subprocess
from pathlib import Path

import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/huffman"
VIDEO = OUT / "videos/scene/480p15/HuffmanEncoding.mp4"


def main():
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(VIDEO)
    ]))
    stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (854, 480, "15/1")
    shots = json.loads((OUT / "shot_times.json").read_text())["shots"]
    targets = [shot["start"] + (8 if "actual Huffman tree" in shot["title"] else 4) for shot in shots]
    next_target = 0
    decoded = 0
    pictures = []
    with av.open(str(VIDEO)) as container:
        for frame in container.decode(video=0):
            decoded += 1
            if next_target < len(targets) and float(frame.time) >= targets[next_target]:
                picture = frame.to_image()
                picture.save(OUT / f"inspection-{next_target + 1:02d}.png")
                pictures.append(picture)
                next_target += 1
    assert len(pictures) == len(shots)
    sheet = Image.new("RGB", (427 * 3, 274 * 4), "#202938")
    draw = ImageDraw.Draw(sheet)
    for i, (picture, shot) in enumerate(zip(pictures, shots)):
        x, y = (i % 3) * 427, (i // 3) * 274
        draw.text((x + 8, y + 6), f"{targets[i]:.1f}s  {shot['title'][:49]}", fill="white")
        sheet.paste(picture.resize((427, 240)), (x, y + 28))
    sheet.save(OUT / "contact-sheet.png")
    result = {"path": str(VIDEO), "bytes": VIDEO.stat().st_size,
              "duration": float(metadata["format"]["duration"]), "decoded_frames": decoded,
              "resolution": [stream["width"], stream["height"]], "fps": stream["r_frame_rate"],
              "codec": stream["codec_name"], "sampled_shots": len(shots),
              "note": "Full decode and per-shot samples, not a claim of subjective playback approval."}
    (OUT / "inspection-report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
