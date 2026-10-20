#!/usr/bin/env python3
"""Assemble the Hugging Face repository in build/hf/ (nothing is uploaded).

Copies the exported graphs (build/onnx/), the model card, NOTICE and license
texts (huggingface/), and writes manifest.json with the sha256, size and
signature of every file and the upstream revisions. Prints the file list.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import onnx

from meanvc2_runtime import MEANVC2_MODEL_REVISION, MEANVC2_REVISION, S3PRL_REVISION, WAVLM_MODEL_REVISION

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "build/onnx"
OUT = REPO / "build/hf"
FILES = {  # name in the HF repo -> exported file
    "fbank.onnx": "fbank.onnx", "asr.onnx": "asr.onnx", "asr.int8.onnx": "asr.int8.onnx", "gtm.onnx": "gtm.onnx",
    "dit.onnx": "dit.onnx", "dit.int8.onnx": "dit.int8.onnx", "vocos.onnx": "vocos.onnx", "vocos.int8.onnx": "vocos.int8.onnx",
    "speaker/speaker.onnx": "speaker.onnx", "speaker/speaker.int8.onnx": "speaker.int8.onnx",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def signature(path: Path) -> dict:
    m = onnx.load(str(path), load_external_data=False)
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    return {"inputs": {i.name: dims(i) for i in m.graph.input}, "outputs": {o.name: dims(o) for o in m.graph.output},
            "opset": max(o.version for o in m.opset_import)}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "speaker").mkdir(parents=True)
    manifest = {"model": "MeanVC2 120ms_40ms, streaming", "sampleRate": 16000, "chunkSamples": 2560,
                "upstream": {"code": f"https://github.com/ASLP-lab/MeanVC2@{MEANVC2_REVISION}",
                             "weights": f"https://huggingface.co/ASLP-lab/MeanVC2@{MEANVC2_MODEL_REVISION}",
                             "wavlm": f"https://huggingface.co/s3prl/converted_ckpts@{WAVLM_MODEL_REVISION}",
                             "s3prl": f"https://github.com/s3prl/s3prl@{S3PRL_REVISION}"},
                "licenses": {"speaker/": "CC-BY-SA-3.0", "*": "Apache-2.0"}, "files": {}}
    for name, src in FILES.items():
        dst = OUT / name
        shutil.copy2(SRC / src, dst)
        manifest["files"][name] = {"bytes": dst.stat().st_size, "sha256": sha256(dst), **signature(dst)}
    for f in ("README.md", "NOTICE"):
        shutil.copy2(REPO / "huggingface" / f, OUT / f)
    shutil.copytree(REPO / "huggingface/licenses", OUT / "licenses")
    shutil.copy2(REPO / "huggingface/licenses/CC-BY-SA-3.0-UniSpeech.txt", OUT / "speaker/LICENSE")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(OUT / "manifest.json", REPO / "huggingface/manifest.json")
    total = 0
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            print(f"{p.stat().st_size:>14,}  {p.relative_to(OUT)}")
    print(f"{total:>14,}  total")


if __name__ == "__main__":
    main()
