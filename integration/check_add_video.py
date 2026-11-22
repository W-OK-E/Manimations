"""Prove that adding a clip needs only JSON/media, in an ignored isolated copy."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "build/add-video-qa"
COPY = QA / "source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=COPY)


def main():
    COPY.mkdir(parents=True, exist_ok=True)
    for name in ("catalog.py", "render.py", "stage_pages.py"):
        target = COPY / "scripts" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    for name in ("index.html", "gallery.css", "gallery.js"):
        target = COPY / "docs" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "docs" / name, target)
    frontend_before = {name: digest(COPY / "docs" / name) for name in ("index.html", "gallery.css", "gallery.js")}
    catalog = COPY / "animations/catalog.json"
    catalog.parent.mkdir(exist_ok=True)
    entries = [{"slug": "first-sample", "title": "First sample", "tags": ["Existing topic"],
                "description": "Isolated QA entry using the approved playback test."}]
    catalog.write_text(json.dumps(entries))
    input_video = ROOT / "docs/media/playback-test.mp4"
    run(sys.executable, COPY / "scripts/render.py", "first-sample", "--input", input_video)
    first_hash = digest(COPY / "docs/media/first-sample.mp4")
    # This is the user's daily operation: append metadata, supply a video, export.
    entries.append({"slug": "second-sample", "title": "Second sample", "tags": ["New topic"],
                    "description": "Added using JSON and assets only; no website code was changed."})
    catalog.write_text(json.dumps(entries))
    run(sys.executable, COPY / "scripts/render.py", "second-sample", "--input", input_video)
    manifest = json.loads((COPY / "docs/animations.json").read_text())
    assert [item["slug"] for item in manifest["items"]] == ["first-sample", "second-sample"]
    assert digest(COPY / "docs/media/first-sample.mp4") == first_hash
    second_hash = digest(COPY / "docs/media/second-sample.mp4")
    # Metadata-only edits must not re-encode either MP4.
    entries[1]["title"] = "Updated second title"
    catalog.write_text(json.dumps(entries))
    run(sys.executable, COPY / "scripts/catalog.py")
    assert json.loads((COPY / "docs/animations.json").read_text())["items"][1]["title"] == "Updated second title"
    assert digest(COPY / "docs/media/second-sample.mp4") == second_hash
    assert frontend_before == {name: digest(COPY / "docs" / name) for name in frontend_before}
    config = QA / "pages-test-config.json"
    config.write_text(json.dumps({"release_tag": "local-qa-only", "approved_slugs": [entry["slug"] for entry in entries]}))
    # Fresh destination per run; no cleanup/deletion of earlier review artifacts.
    hosts = QA / "hosts"
    hosts.mkdir(exist_ok=True)
    number = 1
    while (hosts / f"run-{number}").exists():
        number += 1
    destination = hosts / f"run-{number}" / "Manimations"
    run(sys.executable, COPY / "scripts/stage_pages.py", "--config", config,
        "--media-dir", COPY / "docs/media", "--destination", destination)
    final = json.loads((destination / "animations.json").read_text())
    assert [item["slug"] for item in final["items"]] == [entry["slug"] for entry in entries]
    report = {"passed": True, "added_via_json_and_assets_only": True,
              "frontend_hashes_unchanged": True, "existing_video_unchanged": True,
              "metadata_edit_did_not_reencode": True, "manim_required_for_import": False,
              "staged_directory": str(destination), "host_directory": str(destination.parent)}
    (QA / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
