#!/usr/bin/env python3
"""Fixtures for the example app's self-check (build/fixtures/, not committed).

- target.f32: the reference voice; spk.f32: its embedding by the upstream
  encoder (PyTorch), so the app can compare its own speaker encoder
- source.f32: the voice to convert, padded to whole 160 ms chunks
- noise.f32: the ODE noise the pipeline draws, [1, 16, 80] per DiT step
- expected_{fp32,int8}_s3.f32: tool/onnx_stream.py (Python ONNX Runtime) on
  the same source, embedding and noise, 3 steps

Without --source/--reference, two macOS `say` voices are used.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from onnx_stream import CHUNK, VARIANTS, OnnxStreamingVC

REPO = Path(__file__).resolve().parents[1]


def say(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "say.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEI16@16000", text],
                       check=True)
        audio, _ = sf.read(path, dtype="float32")
    return audio


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--models", type=Path, default=REPO / "build/onnx")
    p.add_argument("--out", type=Path, default=REPO / "build/fixtures")
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    source = load(a.source) if a.source else say("Kyoko", "こんにちは。今日はいい天気ですね。少し散歩してから、駅前のカフェでコーヒーを飲みませんか？")
    target = load(a.reference) if a.reference else say("Samantha", "The quick brown fox jumps over the lazy dog. I would like a cup of coffee, please.")
    n = -(-len(source) // CHUNK)
    source = np.pad(source, (0, n * CHUNK - len(source))).astype(np.float32)

    from stream_vc import StreamingVC
    vc = StreamingVC(threads=4, model="120ms", steps=3)
    vc.set_target(target)
    spk = vc.runner.vc_spk_emb.numpy().astype(np.float32).ravel()

    rng = np.random.default_rng(a.seed)
    draws = [rng.standard_normal((1, 16, 80), dtype=np.float32) for _ in range(n * 4)]
    out = {"target.f32": target, "spk.f32": spk, "source.f32": source, "noise.f32": np.concatenate(draws).ravel()}
    for variant in ("fp32", "int8"):
        ox = OnnxStreamingVC(a.models, suffix=VARIANTS[variant], threads=1, steps=3)
        ox.set_target(spk)
        it = iter(draws)
        ox.noise_fn = lambda k: next(it)
        out[f"expected_{variant}_s3.f32"] = np.concatenate([ox.process(source[i:i + CHUNK]) for i in range(0, len(source), CHUNK)])
    a.out.mkdir(parents=True, exist_ok=True)
    for name, arr in out.items():
        np.asarray(arr, "<f4").tofile(a.out / name)
        print(name, arr.size)


if __name__ == "__main__":
    main()
