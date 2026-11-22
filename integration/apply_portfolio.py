"""Install only new Jekyll integration files; refuse to overwrite existing work."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    args = parser.parse_args()
    portfolio = args.portfolio.resolve()
    branch = subprocess.check_output(["git", "-C", str(portfolio), "branch", "--show-current"], text=True).strip()
    if branch in ("main", "master"):
        parser.error("Switch the portfolio to a local integration branch first")
    source = Path(__file__).resolve().parent / "portfolio"
    files = sorted(path for path in source.rglob("*") if path.is_file())
    targets = [(path, portfolio / path.relative_to(source)) for path in files]
    if any(target.exists() for _, target in targets):
        parser.error("An integration file already exists; review it rather than overwrite it")
    for path, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text())
        print(target)


if __name__ == "__main__":
    main()
