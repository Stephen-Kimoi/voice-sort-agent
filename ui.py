"""Gradio demo: upload a spoken sort-rule recording, then classify item images live.

Speechmatics' realtime client is async; Gradio's callbacks are sync generators.
We bridge the two with asyncio.run() per call rather than a background thread +
queue, because each step here (one short command, one image) completes in under
a couple seconds — a thread/queue bridge only earns its complexity when a
callback needs to yield partial results while a long stream is still running.
"""

import asyncio
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from anomaly_classifier import OpenVINOAnomalyClassifier
from rule_parser import SortRule, parse_rule
from transcriber import transcribe_command

load_dotenv()

MODEL_XML = Path("model/weights/openvino/model.xml")
_classifier: OpenVINOAnomalyClassifier | None = None


def _get_classifier() -> OpenVINOAnomalyClassifier:
    global _classifier
    if _classifier is None:
        if not MODEL_XML.exists():
            raise gr.Error(f"No trained model at {MODEL_XML}. Run `python anomaly_classifier.py` first.")
        _classifier = OpenVINOAnomalyClassifier(MODEL_XML)
    return _classifier


def set_rule(audio_path: str | None):
    if not audio_path:
        raise gr.Error("Record or upload a spoken sort command first.")
    transcript = asyncio.run(transcribe_command(audio_path))
    rule = parse_rule(transcript)
    rule_summary = f"anomalous items -> bin {rule.anomalous_bin}  |  normal items -> bin {rule.normal_bin}"
    return transcript, rule_summary, rule


def classify_image(image_path: str | None, rule: SortRule | None):
    if not image_path:
        raise gr.Error("Choose an item image to classify.")
    if rule is None:
        raise gr.Error("Set the sort rule from a spoken command first.")
    result = _get_classifier().classify(Path(image_path))
    bin_number = rule.bin_for(result["is_anomalous"])
    verdict = "ANOMALOUS" if result["is_anomalous"] else "normal"
    return (
        f"{result['score']:.3f}",
        verdict,
        f"Bin {bin_number}",
    )


with gr.Blocks(title="Voice-Driven Sort Agent") as demo:
    gr.Markdown(
        "# Voice-Driven Sort Agent\n"
        "1. Record or upload a spoken sort rule (e.g. *\"send anomalous items to bin two, "
        "normal items to bin one\"*).\n"
        "2. Pick an item image — the OpenVINO/Anomalib classifier scores it and the rule "
        "decides which bin it goes to."
    )

    rule_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Set the sort rule")
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Spoken sort command")
            set_rule_btn = gr.Button("Transcribe & set rule")
            transcript_out = gr.Textbox(label="Transcript", interactive=False)
            rule_out = gr.Textbox(label="Parsed rule", interactive=False)

        with gr.Column():
            gr.Markdown("### 2. Classify an item")
            image_in = gr.Image(type="filepath", label="Item image")
            classify_btn = gr.Button("Classify & route")
            score_out = gr.Textbox(label="Anomaly score", interactive=False)
            verdict_out = gr.Textbox(label="Verdict", interactive=False)
            bin_out = gr.Textbox(label="Bin decision", interactive=False)

    set_rule_btn.click(
        fn=set_rule,
        inputs=[audio_in],
        outputs=[transcript_out, rule_out, rule_state],
    )
    classify_btn.click(
        fn=classify_image,
        inputs=[image_in, rule_state],
        outputs=[score_out, verdict_out, bin_out],
    )

if __name__ == "__main__":
    demo.launch()
