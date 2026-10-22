"""Kör befintliga steg fram till validering. Sluttest körs separat."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "load_data.py",
    "prepare_data.py",
    "split_data.py",
    "train_baseline.py",
    "train_model.py",
)


def run_pipeline(download: bool = False) -> None:
    """Använd samma Pythonmiljö och avbryt direkt om ett steg misslyckas."""
    raw_path = PROJECT_ROOT / "data/raw/yelp_labelled.txt"
    if not download and not raw_path.is_file():
        raise FileNotFoundError(
            "Yelp-filen saknas. Kör src/run_pipeline.py med --download "
            "för att hämta originaldatan."
        )
    for script_name in STEPS:
        script = PROJECT_ROOT / "src" / script_name
        if not script.is_file():
            raise FileNotFoundError(f"Projektfil saknas: {script}")
        command = [sys.executable, str(script)]
        if script_name == "load_data.py" and download:
            command.append("--download")
        print(f"\nKör {script_name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(
        "\nFlödet fram till validering är klart. Modeller och rapporter har skapats.\n"
        "Slututvärdering har inte körts. Kör src/final_evaluation.py separat "
        "när modellvalet är låst."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Hämta originaldatan före körning.")
    args = parser.parse_args()
    try:
        run_pipeline(download=args.download)
    except (FileNotFoundError, PermissionError) as error:
        print(f"Fel: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"Flödet avbröts vid {Path(error.cmd[1]).name} "
            f"(felkod {error.returncode}). Åtgärda felet ovan och kör igen.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
