"""Builds sort_command.wav from a spoken sort rule using macOS `say` + `afconvert`.

Only needed to (re)generate the fixture; the tutorial itself just streams the
resulting WAV through Speechmatics. Not portable off macOS — if you're on
another OS, record or supply your own 16kHz mono WAV command instead.
"""

import subprocess
from pathlib import Path

COMMAND_TEXT = "Send anomalous items to bin two, and normal items to bin one."
OUTPUT_PATH = Path("sort_command.wav")
SAMPLE_RATE = 16000
VOICE = "Samantha"


def main() -> None:
    aiff_path = Path("_command.aiff")
    subprocess.run(["say", "-v", VOICE, "-o", str(aiff_path), COMMAND_TEXT], check=True)
    subprocess.run(
        [
            "afconvert",
            str(aiff_path),
            str(OUTPUT_PATH),
            "-d", "LEI16",
            "-c", "1",
            "-r", str(SAMPLE_RATE),
            "-f", "WAVE",
        ],
        check=True,
    )
    aiff_path.unlink()
    print(f"Wrote {OUTPUT_PATH}: \"{COMMAND_TEXT}\"")


if __name__ == "__main__":
    main()
