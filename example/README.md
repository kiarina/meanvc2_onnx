# meanvc2_onnx example

Downloads the models from Hugging Face (or reads `MODEL_DIR`), registers the reference voice with the speaker encoder, converts in 160 ms chunks,
and with `FIXTURE_DIR` (made by `tool/make_check_fixture.py`) compares with the Python ONNX Runtime output for the same noise.

```sh
flutter run -d macos --profile --dart-define=MODEL_DIR=$PWD/../build/hf --dart-define=FIXTURE_DIR=$PWD/../build/fixtures --dart-define=VARIANT=int8
```

Settings: `MODEL_DIR`, `FIXTURE_DIR`, `REVISION` (Hugging Face commit to download), `VARIANT` (`int8`, `fp32`), `ORT_THREADS`, `AUTO_EXIT`.
The result is printed as one `MEANVC2_EXAMPLE {...}` line.
