# Hugging Face と GitHub に公開する

## 状態

- 2026-10-21: 公開の直前まで用意した。オーナーの承認待ち（通話機能のセッション経由で見せる）
- 2026-10-21: モデルカード・NOTICE の本文をオーナーが見てから公開する
- GitHub は private で作った。Hugging Face は未作成。pub.dev は API が落ち着いてから（通話機能に組み込んだ後）

## やること（承認の後）

1. `cd tool && uv run python stage_hf.py` で `build/hf/` を作り直す（中身はこの repo の `huggingface/` と `build/onnx/`）
2. Hugging Face に `kiarina/MeanVC2-ONNX` を作り、`build/hf/` を上げる（`hf upload kiarina/MeanVC2-ONNX build/hf .`）。上げた commit を README とこのファイルに書く
3. GitHub の `kiarina/meanvc2_onnx` を公開にする
4. example を Hugging Face の commit を `REVISION` に入れて動かし、ダウンロードからの通しを確かめる
5. Spirits Garden の通話機能から git の依存（commit 固定）で使う

## 申し送り

- 名前は既存とぶつからないことを確かめた（2026-10-21）: Hugging Face に `kiarina/MeanVC2-ONNX` は無い。似た名前に LavMic の `djihuiyong/lavmic-meanvc2-onnx`（2026-10-18、Android の短い区間の変換、ASR なし、速さ・聴取の主張なし）がある。
  pub.dev・GitHub に `meanvc2_onnx` は無い
- ライセンスの確認結果は `huggingface/NOTICE`。話者の encoder は microsoft/UniSpeech の CC BY-SA 3.0。
- MeanVC2 の GitHub には LICENSE のファイルが無く、README の文と Hugging Face のメタデータが Apache-2.0
