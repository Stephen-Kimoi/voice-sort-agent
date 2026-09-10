"""Streams a WAV file to Speechmatics real-time and returns the final transcript."""

import io
import os
import wave

from speechmatics.rt import (
    AsyncClient,
    AudioFormat,
    AudioEncoding,
    TranscriptionConfig,
    ServerMessageType,
)


async def transcribe_command(wav_path: str, on_partial=None) -> str:
    """Streams wav_path to Speechmatics and returns the concatenated final transcript.

    on_partial, if given, is called with each transcript fragment as it arrives —
    used by the Gradio UI to show transcription happening live rather than as a
    single blocking call.
    """
    api_key = os.environ["SPEECHMATICS_API_KEY"]

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        raw_pcm = wf.readframes(wf.getnframes())

    client = AsyncClient(api_key=api_key)
    fragments: list[str] = []

    @client.on(ServerMessageType.ADD_TRANSCRIPT)
    def _on_final_transcript(message):
        text = message.get("metadata", {}).get("transcript", "").strip()
        if text:
            fragments.append(text)
            if on_partial:
                on_partial(text)

    audio_format = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate)
    transcription_config = TranscriptionConfig(language="en", enable_partials=False)

    async with client:
        await client.transcribe(
            io.BytesIO(raw_pcm),
            transcription_config=transcription_config,
            audio_format=audio_format,
        )

    return " ".join(fragments).strip()
