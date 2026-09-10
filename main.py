"""End-to-end pipeline: spoken sort rule + item images -> per-item bin decisions.

Run order: transcribe the spoken command -> parse it into a SortRule -> load the
trained OpenVINO anomaly classifier -> classify every image under data/test/ ->
fuse each classification with the rule to print a bin decision. This mirrors
what a real sorting line would do once per shift (set the rule) and then per
item (classify + route) — the CLI just runs all the items in one pass.
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from anomaly_classifier import OpenVINOAnomalyClassifier
from rule_parser import parse_rule
from transcriber import transcribe_command

MODEL_XML = Path("model/weights/openvino/model.xml")
COMMAND_WAV = Path("sort_command.wav")
TEST_IMAGE_DIRS = (Path("data/test/good"), Path("data/test/anomalous"))


async def run() -> None:
    print(f"Transcribing {COMMAND_WAV} ...")
    transcript = await transcribe_command(str(COMMAND_WAV))
    print(f"Transcript: \"{transcript}\"")

    rule = parse_rule(transcript)
    print(f"Parsed rule: anomalous -> bin {rule.anomalous_bin}, normal -> bin {rule.normal_bin}\n")

    if not MODEL_XML.exists():
        raise SystemExit(
            f"No trained model at {MODEL_XML}. Run `python anomaly_classifier.py` first."
        )
    classifier = OpenVINOAnomalyClassifier(MODEL_XML)

    print(f"{'image':<32} {'score':>7}  {'call':<10} bin")
    for directory in TEST_IMAGE_DIRS:
        for image_path in sorted(directory.glob("*.png")):
            result = classifier.classify(image_path)
            bin_number = rule.bin_for(result["is_anomalous"])
            call = "anomalous" if result["is_anomalous"] else "normal"
            print(f"{image_path.name:<32} {result['score']:>7.3f}  {call:<10} {bin_number}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(run())
