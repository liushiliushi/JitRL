"""Replace the website placeholders with website domains from env_config
Generate the test data"""

import os
import json
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    in_path = script_dir / "test.raw.json"
    out_dir = (script_dir / ".." / "config_files").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", os.environ.get("WA_GITLAB"))
    raw = raw.replace("__REDDIT__", os.environ.get("WA_REDDIT"))
    raw = raw.replace("__SHOPPING__", os.environ.get("WA_SHOPPING"))
    raw = raw.replace("__SHOPPING_ADMIN__", os.environ.get("WA_SHOPPING_ADMIN"))
    raw = raw.replace("__WIKIPEDIA__", os.environ.get("WA_WIKIPEDIA"))
    raw = raw.replace("__MAP__", os.environ.get("WA_MAP"))
    with open(out_dir / "test.json", "w") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(out_dir / f"{idx}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()