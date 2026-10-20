---
license: other
license_name: apache-2.0-and-cc-by-sa-3.0
license_link: https://huggingface.co/kiarina/MeanVC2-ONNX/blob/main/NOTICE
base_model:
- ASLP-lab/MeanVC2
pipeline_tag: audio-to-audio
library_name: onnx
tags:
- onnx
- voice-conversion
- zero-shot
- streaming
- meanvc2
---

# MeanVC2-ONNX

[MeanVC2](https://github.com/ASLP-lab/MeanVC2) (ASLP@NPU) streaming zero-shot voice conversion, exported to ONNX so it runs in 160 ms chunks
with ONNX Runtime alone — no PyTorch, no Python. The streaming state (the ASR caches and the DiT's per-layer KV cache) is an input and an output
of each graph, so the caller keeps it between chunks.

Everything the upstream `runtime/run_rt.py` streaming path needs is here, including the feature extraction and the ASR encoder,
plus the speaker encoder for registering a new reference voice.

- Dart / Flutter: [kiarina/meanvc2_onnx](https://github.com/kiarina/meanvc2_onnx) (the host loop, a model downloader and the speaker encoder)
- Python reference of the same host loop, the exporters and the checks: `tool/` in the same repository

## Files

`120ms` model (120 ms chunk + 40 ms look-ahead), 16 kHz.

| File | What | fp32 | int8 |
|---|---|---:|---:|
| `fbank.onnx` | 400-sample frames → Kaldi log-mel [N, 80] | 0.9 MB | (fp32 only) |
| `asr.onnx` | Fast-U2++ 160 ms: one 19-frame window + offset + caches → BN [1, 4, 256] + new caches | 75.8 MB | 28.7 MB |
| `gtm.onnx` | speaker embedding [1, 256] → timbre memory k, v [1, 32, 256] (once per voice) | 17.4 MB | (fp32 only) |
| `dit.onnx` | one mean-flow DiT evaluation on [1, 16, 80] + KV caches (length 0–36) → u + new caches | 53.6 MB | 13.8 MB |
| `vocos.onnx` | mel [1, 80, T] → waveform [1, 160 (T − 1)] | 36.5 MB | 13.0 MB |
| `speaker/speaker.onnx` | reference waveform [1, T] → speaker embedding [1, 256] (**CC BY-SA 3.0**) | 1.30 GB | 387 MB |

During a call you need `fbank`, `asr`, `dit` and `vocos`: 56 MB with int8 (what we use by default), 167 MB in fp32.
`gtm` and the speaker encoder are only for setting the target voice; the 256 numbers of an embedding can be made once and kept.
`manifest.json` has the sha256 and size of every file and the exact upstream revisions. Pin a commit in URLs (`/resolve/<commit>/...`).

## How it matches upstream

We fed the same audio in 160 ms chunks to the upstream PyTorch streaming code and to these graphs, with the same ODE noise
(numpy-generated and injected on both sides), and compared each stage.

- fp32, Python ONNX Runtime 1.31 vs PyTorch 2.5.1: waveform SNR 38–67 dB, log-spectral distance 0.01–0.06 dB, over three inputs
  (6.6 s, 7.9 s and a 70 s input that crosses upstream's position reset). Running upstream itself with another noise seed differs by about 13 dB LSD,
  so the export is far below what changes between two upstream runs.
- The same fp32 graphs inside a Flutter app (flutter_onnxruntime 1.9.0, ONNX Runtime 1.28) on macOS, Windows, Android and iOS:
  SNR 63 dB and LSD 0.02 dB against upstream PyTorch.
- int8: about 5.7 dB LSD against upstream, and int8 results also move between ONNX Runtime versions (about 5 dB between 1.28 and 1.31).
  In a blind-capable side-by-side listening page, the one listener we had (the author) could hardly tell int8 from PyTorch.
  This is one listener, so listen on your own material before relying on it, and pin your ONNX Runtime version.
- Speaker encoder: fp32 cosine 1.0 against upstream (3–20 s references); int8 0.9993–0.9997 on speech references.

The full method, numbers and the listening page are in the lab:
[labs/2026/10/21/meanvc2-onnx-streaming](https://github.com/kiarina/labs/tree/main/2026/10/21/meanvc2-onnx-streaming).

## Speed

Processing time per 160 ms chunk, 3 ODE steps, inside a Flutter app, back to back (median / 95th percentile, best thread count):

| Device | fp32 | int8 |
|---|---:|---:|
| MacBook Pro, Apple M1 Max | 19 / 36 ms | 14 / 24 ms |
| Core i9-12900K, Windows 11 | 25 / 44 ms | 15 / 27 ms |
| iPad Air 4 (A14) | 29 / 52 ms | 18 / 32 ms |
| Pixel 6 (Tensor G1) | 69 / 128 ms | 54 / 97 ms |

On the Pixel 6, paced in real time for 5 minutes (2 threads), int8 averaged 92 ms per chunk (worst 188 ms) and fp32 103 ms (worst 220 ms);
both kept up, but there is not much room left for anything else. The fastest thread count differs by device (the Pixel 6 is slower with 1 or 4 threads than with 2),
so measure it on the device. The first output comes after about 330–400 ms (upstream behaves the same).
The upstream PyTorch path on the same Mac takes 38 ms per chunk with one thread.

## Using it

The host loop is a line-by-line port of upstream `VCRunner.process_chunk`: cut 400-sample frames every 160 samples, slide the 19-frame ASR window
by 16, upsample the BN features to 16 frames per call, run the DiT 2–3 times per 12 output frames with the noise of the overlapping 4 frames carried over,
and cross-fade the vocoder output. The Dart package and `tool/onnx_stream.py` both do exactly this; start from one of them rather than from the graph signatures.

## Licenses

- Conversion graphs: **Apache-2.0**, from the MeanVC2 weights released under Apache-2.0. Vocos architecture MIT, WeNet Apache-2.0.
- Speaker encoder: **CC BY-SA 3.0**. It is Microsoft's WavLM-Large + ECAPA-TDNN speaker verification checkpoint from microsoft/UniSpeech, which is
  CC BY-SA 3.0; the ONNX files are a modified version and keep that license (WavLM itself is MIT).

Details and attributions: [NOTICE](NOTICE), license texts in `licenses/`. Not affiliated with or endorsed by the MeanVC2 authors.

## Responsible use

This converts anyone's speech into a voice it has heard for a few seconds. Only use reference voices with the consent of the person they belong to,
and do not use it to impersonate people or to make audio that misleads others about who is speaking. The MeanVC2 authors ask the same.

## Citation

Please cite MeanVC2:

```bibtex
@article{ma2026meanvc2,
  title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2606.09050},
  year={2026}
}
```
