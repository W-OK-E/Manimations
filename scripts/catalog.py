"""Publish bounded static manifest pages; the browser never fetches the whole catalog."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE_SIZE = 12


def read_source(path=ROOT / "animations/catalog.json"):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def publish(items, destination=ROOT / "docs"):
    """Write only indexes, never render scenes, copy media, or remove existing files."""
    destination = Path(destination)
    seen = set()
    for item in items:
        slug = item["slug"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in seen:
            raise ValueError(f"Invalid or duplicate slug: {slug}")
        seen.add(slug)
        for field in ("title", "description", "poster", "video"):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(f"Missing {field} for {slug}")
        if not isinstance(item["tags"], list) or not all(isinstance(tag, str) for tag in item["tags"]):
            raise ValueError(f"Invalid tags for {slug}")
        if not isinstance(item["duration"], (int, float)) or item["duration"] <= 0:
            raise ValueError(f"Invalid duration for {slug}")
        for field in ("poster", "video"):
            if not item[field].startswith("media/") or ".." in item[field] or ":" in item[field]:
                raise ValueError(f"Expected local media path for {slug}")
    pages = [items[i:i + PAGE_SIZE] for i in range(0, len(items), PAGE_SIZE)] or [[]]
    for number, page in enumerate(pages):
        name = "animations.json" if number == 0 else f"catalog/page-{number + 1}.json"
        next_page = f"catalog/page-{number + 2}.json" if number + 1 < len(pages) else None
        write_json(destination / name, {"items": page, "next": next_page})
    for item in items:
        write_json(destination / f"catalog/{item['slug']}.json", item)


def main(destination=ROOT / "docs"):
    destination = Path(destination)
    items = []
    for entry in read_source():
        metadata = destination / "media" / f"{entry['slug']}.json"
        if metadata.exists():
            result = json.loads(metadata.read_text())
            result.update(entry)
            items.append(result)
    publish(items, destination)
    print(f"Published {len(items)} entries in pages of at most {PAGE_SIZE}.")


if __name__ == "__main__":
    main()
