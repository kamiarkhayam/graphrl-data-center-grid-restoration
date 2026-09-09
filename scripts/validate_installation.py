"""Import the package and perform one tiny deterministic fixture rollout."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    os.environ["DC_RESTORATION_ROOT"] = str(args.root.resolve())
    from dc_restoration.installation import validate

    print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()
