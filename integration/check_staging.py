"""Integration checks for the real staged artifact and deliberate TODO gates."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stage = ROOT / "build/integration/host/Manimations"
    manifest = json.loads((stage / "animations.json").read_text())
    assert [item["slug"] for item in manifest["items"]] == ["huffman"]
    assert manifest["next"] is None
    assert not (stage / "media/playback-test.mp4").exists()
    assert not (stage / "catalog/dummy-slug.json").exists()
    assert (stage / "catalog/huffman.json").is_file()
    assert (stage / ".nojekyll").is_file()
    for item in manifest["items"]:
        for key in ("video", "poster"):
            assert (stage / item[key]).is_file()
    pending = subprocess.run([sys.executable, str(ROOT / "scripts/stage_pages.py"),
                              "--write-plan", str(ROOT / "build/integration/should-not-exist.json")],
                             cwd=ROOT, text=True, capture_output=True)
    assert pending.returncode == 1 and "TODO(om)" in pending.stderr
    assert not (ROOT / "build/integration/should-not-exist.json").exists()
    reused = subprocess.run([sys.executable, str(ROOT / "scripts/stage_pages.py"),
                             "--config", str(ROOT / "build/integration/pages-test-config.json"),
                             "--media-dir", str(ROOT / "build/github-pages-media-review/media"),
                             "--destination", str(stage)], cwd=ROOT, text=True, capture_output=True)
    assert reused.returncode == 1 and "new/empty" in reused.stderr
    config = json.loads((ROOT / "integration/pages.json").read_text())
    assert config["release_tag"] == "" and config["approved_slugs"] == []
    print("PASS: selected assets only; subpath manifests/media complete; unfilled TODO blocks; no stale-directory reuse; real selection unchanged.")


if __name__ == "__main__":
    main()
