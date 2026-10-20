# meanvc2_onnx

Streaming zero-shot voice conversion with [MeanVC2](https://github.com/ASLP-lab/MeanVC2), on ONNX Runtime, in Flutter.
Feed 160 ms of 16 kHz audio at a time and get the same speech back in another voice, about 0.35 s later.
No PyTorch and no Python at run time; the models are the ONNX exports in
[kiarina/MeanVC2-ONNX](https://huggingface.co/kiarina/MeanVC2-ONNX).

It is the upstream `runtime/run_rt.py` streaming loop ported to Dart, line by line, so the output matches the upstream PyTorch code
(log-spectral distance 0.02 dB with fp32 on macOS, Windows, Android and iOS). A new target voice is registered with the bundled speaker encoder
from a few seconds of speech.

> Status: 0.1.0-dev, used as a git dependency while the API settles. Not on pub.dev yet.

## Use

```yaml
dependencies:
  meanvc2_onnx:
    git:
      url: https://github.com/kiarina/meanvc2_onnx
      ref: <commit>
```

```dart
import 'package:meanvc2_onnx/meanvc2_onnx.dart';

// 1. models (the Hugging Face commit this package version was checked against,
//    MeanVc2Models.defaultRevision, unless you pass `revision:`)
await MeanVc2Models.download(dir, files: MeanVc2Models.files(ModelVariant.int8, speaker: ModelVariant.int8));

// 2. register a voice once, keep the 256 numbers
final encoder = await SpeakerEncoder.load('$dir/${MeanVc2Models.speakerFileName(ModelVariant.int8)}');
final embedding = await encoder.embed(reference16k); // Float32List, mono 16 kHz
await encoder.close();

// 3. convert while the audio streams in
final vc = await MeanVc2Stream.load(dir, variant: ModelVariant.int8, threads: 2);
await vc.setTarget(embedding);
final out = await vc.process(chunk160ms); // Float32List, may be empty at first
```

`process` holds back what it cannot convert yet; call `reset()` between utterances that should not be joined.
The output can go above 1.0, so level it before writing 16-bit audio.

## Which variant, how many threads

- **int8** (default): 56 MB for conversion. In side-by-side listening against upstream PyTorch the difference was hard to hear (one listener),
  and it is faster. int8 results shift with the ONNX Runtime version, so pin it and listen again when you upgrade.
- **fp32**: 167 MB, numerically the same as upstream. Use it as a reference or if int8 sounds off on your material.
- **Threads**: measure on the device. The fastest count was 2 on a Mac, a Windows PC and a Pixel 6, and 1–2 on an iPad Air 4; a Pixel 6 with 1 or 4 threads
  missed the 160 ms deadline often.

Per 160 ms chunk, 3 ODE steps, median: Mac (M1 Max) 19 ms fp32 / 14 ms int8, Windows (i9-12900K) 25 / 15, iPad Air 4 29 / 18, Pixel 6 69 / 54.
On a Pixel 6 paced in real time for 5 minutes, int8 averaged 92 ms (worst 188 ms). That keeps up, but leaves little room for other work,
so on slow devices measure under real load and fall back to the unconverted voice when it does not keep up.

## Playing the output in real time

- The output comes in bursts: `process` returns nothing for the first chunks (the first audio comes about 330–400 ms after the first input)
  and then gives back 0, 1 or 2 chunks' worth per call, while the input keeps arriving at 160 ms a chunk.
- So keep headroom in the playback queue: one chunk plus 60–100 ms (we buffer 240 ms at 48 kHz before playing). With only a few tens of milliseconds
  the queue runs dry now and then even on a fast machine, and the voice stutters.
- If you fall back to the unconverted voice when conversion does not keep up, count the playback running short only while there is voice to play
  (not during silence between utterances, nor while waiting for the first output), over a window of several seconds rather than per chunk.

## Speaker encoder license

`speaker/speaker*.onnx` is Microsoft's WavLM-Large + ECAPA-TDNN speaker verification model and is **CC BY-SA 3.0**, unlike the Apache-2.0 conversion graphs.
If you do not want to ship it, make the embeddings elsewhere (for example with upstream's Python code) and pass the 256 numbers to `setTarget`.
See [NOTICE](https://huggingface.co/kiarina/MeanVC2-ONNX/blob/main/NOTICE) in the model repository.

Only use voices with the consent of the people they belong to.

## Repository

- `lib/`: the Dart package
- `example/`: a Flutter app that downloads the models, registers a voice and converts; with `FIXTURE_DIR` it checks itself against the Python output
- `tool/`: the exporters (`export_onnx.py`, `export_speaker.py`), the Python reference of the host loop (`onnx_stream.py`),
  the check fixtures (`make_check_fixture.py`) and the Hugging Face staging (`stage_hf.py`)
- `huggingface/`: the model card, NOTICE, license texts and `manifest.json` of the model repository

```sh
cd tool && uv sync
uv run python export_onnx.py --int8      # fetches MeanVC2 (about 3 GB, hash-checked) into tool/vendor/, writes build/onnx/
uv run python export_speaker.py
uv run python stage_hf.py                # build/hf/: what goes to Hugging Face
uv run python make_check_fixture.py      # build/fixtures/ (uses macOS `say` without --source/--reference)
cd ../example && flutter run -d macos --dart-define=MODEL_DIR=$PWD/../build/hf --dart-define=FIXTURE_DIR=$PWD/../build/fixtures
```

The measurements and the reasoning behind them are in
[labs/2026/10/21/meanvc2-onnx-streaming](https://github.com/kiarina/labs/tree/main/2026/10/21/meanvc2-onnx-streaming).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). MeanVC2 is by ASLP@NPU and collaborators; please cite their paper.
