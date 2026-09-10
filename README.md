# Voice Sort Agent

A software decision pipeline for voice-driven industrial sorting — no robot arm required.

Say a sort rule out loud ("send anomalous items to bin two, normal items to bin one"),
[Speechmatics](https://www.speechmatics.com/) transcribes it in real time, and a
deterministic parser turns it into a structured rule. Separately, an
[Anomalib](https://github.com/open-edge-platform/anomalib) Padim model — exported to
[OpenVINO](https://github.com/openvinotoolkit/openvino) IR and run through OpenVINO's
runtime — classifies an item image as normal or anomalous. The two fuse into a single
per-item bin decision. A Gradio UI shows both halves of the pipeline live: transcript →
parsed rule on one side, image → anomaly score → bin on the other.

This is the companion repo for a Lablab tutorial. It's the software brain only: swap the
printed/UI bin decision for a call to whatever actuator (a Dobot arm, a PLC, a conveyor
gate) you're actually sorting with.

## What's here and why

| File | Why it exists |
|---|---|
| `dataset.py` | Generates a small synthetic "washer" image set (clean ring = normal, cracked/chipped/scratched ring = anomalous). Used instead of the standard MVTec AD dataset — see **Dataset note** below. |
| `anomaly_classifier.py` | Trains a one-class Padim model on the normal images only, exports it to OpenVINO IR, and serves inference through the OpenVINO runtime. |
| `transcriber.py` | Streams a WAV file to Speechmatics' realtime API and returns the transcript. |
| `rule_parser.py` | Deterministic keyword/number parser that turns a transcript into a `SortRule` (which bin gets normal items, which gets anomalous ones). |
| `generate_sample_audio.py` | macOS-only helper (`say` + `afconvert`) that generates `sort_command.wav` from a text script. Not needed if you supply your own 16kHz mono WAV command. |
| `main.py` | CLI: transcribe → parse rule → classify every sample image → print bin decisions. |
| `ui.py` | Gradio demo: record/upload a spoken rule, then classify item images one at a time, with every stage of the pipeline visible. |
| `sample_input.json` | Reference sample command + a couple of test image paths for quick manual checks. |

## Dataset note

Anomalib's standard demo path downloads the *entire* MVTec AD archive (~5.2GB, one file for
all 15 categories) just to reach a single category folder, and that dataset is licensed for
non-commercial research use only. Neither is a good fit for a public tutorial repo, so this
project generates its own small, fully original "washer" image set instead
(`dataset.py`) — no download, no license restriction. Swap in your own product photos for a
real deployment; Padim only needs "normal" examples to train.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in SPEECHMATICS_API_KEY

python dataset.py             # generates data/train and data/test
python anomaly_classifier.py  # trains Padim, exports to model/weights/openvino/model.xml
python generate_sample_audio.py   # macOS only — or supply your own sort_command.wav
```

## Run

```bash
python main.py     # CLI: transcribe sort_command.wav, classify every sample image, print bin decisions
python ui.py        # Gradio demo at http://localhost:7860
```

## Sample output

```
Transcript: "Send anomalous items to bin two and normal items to bin one."
Parsed rule: anomalous -> bin 2, normal -> bin 1

image                              score  call       bin
good_000.png                       0.000  normal     1
anomalous_001.png                  1.000  anomalous  2
anomalous_008.png                  0.333  normal     1   <- borderline, see Gotchas
```

## Gotchas

- **Borderline scores**: with only ~40 synthetic training images and 1 training epoch, Padim's
  decision boundary isn't perfectly sharp — some subtle synthetic defects score close to the
  0.5 threshold and get misclassified. More training images and/or a tuned threshold fix this;
  it's left as-is here to keep the demo fast to build and to keep the pipeline's behavior (not
  a perfectly tuned model) as the point of the tutorial.
- **OpenVINO export path**: Anomalib's `Engine.export(..., export_type="openvino")` writes to
  `model/weights/openvino/model.xml` — the classifier loads that exact path.
- **Speechmatics realtime** needs 16-bit PCM mono WAV; if you record your own command, make
  sure it matches that format (or resample first).
