"""Export the exact public runtime into a new empty directory."""

import argparse
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "public_demo/app.py": "app.py",
    "public_demo/results.json": "results.json",
    "public_demo/requirements.txt": "requirements.txt",
    "public_demo/.streamlit/config.toml": ".streamlit/config.toml",
    "src/dashboard_v3/ui.py": "src/dashboard_v3/ui.py",
}


def package(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a new empty output directory.")
    for source, name in FILES.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / source, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    package(parser.parse_args().output)
