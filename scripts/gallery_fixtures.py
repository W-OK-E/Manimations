"""Create ignored 5/100/1000-entry QA galleries; never modify the real manifest."""

import json
import shutil
from pathlib import Path

from catalog import ROOT, publish


def main():
    original = json.loads((ROOT / "docs/animations.json").read_text())["items"][0]
    sizes = {}
    for count in (1, 5, 100, 1000):
        destination = ROOT / "build/gallery-qa" / str(count) / "Manimations"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("index.html", "gallery.css", "gallery.js"):
            shutil.copy2(ROOT / "docs" / name, destination / name)
        shutil.copytree(ROOT / "docs/fonts", destination / "fonts", dirs_exist_ok=True)
        shutil.copytree(ROOT / "docs/media", destination / "media", dirs_exist_ok=True)
        items = [dict(original, slug=f"sample-{i:04d}", title=f"Playback sample {i + 1}",
                      tags=["Compression" if i % 2 == 0 else "Technical preview"],
                      description="Temporary QA entry using the approved playback-test clip. Not a published mathematical animation.")
                 for i in range(count)]
        publish(items, destination)
        initial = sum((destination / name).stat().st_size for name in ("index.html", "gallery.css", "gallery.js", "animations.json"))
        assert initial < 50_000, initial
        sizes[count] = initial
    assert sizes[100] == sizes[1000], sizes
    report = {"uncompressed_initial_html_css_js_json_bytes": sizes,
              "page_size": 12, "note": "Autoplay adds visible video traffic; posters and videos excluded above."}
    (ROOT / "build/gallery-qa/payload-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
