# Hugging Face と GitHub に公開する

## 状態

- 2026-10-21: 公開した。GitHub https://github.com/kiarina/meanvc2_onnx （public、公開の前に履歴を 1 つの commit に作り直した）、
  Hugging Face https://huggingface.co/kiarina/MeanVC2-ONNX （commit `2db97430c6e315a819daa30e255e14615205a3d1`、19 ファイル）
- package の既定の revision（`MeanVc2Models.defaultRevision`）をその commit にした。example を macOS でダウンロードから動かし、int8・fp32 とも手元の書き出しと同じ結果
  （int8: 話者の encoder のコサイン 0.9996、1 回 14.6 ms／fp32: 1.0、19.8 ms、Python と 53.9 dB）
- pub.dev: 0.1.0 を用意した（2026-10-21）。通話機能が git の依存（commit 固定）で使って動いた API のまま。`.pubignore` で `tool/`・`huggingface/`・`results/`・`tasks/`・`AGENTS.md`・build の生成物を外し、dry-run は警告 0（238 KB）
- pub.dev: 0.1.0 を公開した（2026-10-21、オーナーの承認）。https://pub.dev/packages/meanvc2_onnx 、tag `v0.1.0` を push

## 残り

1. Spirits Garden の通話機能から git の依存（commit 固定）で使う（通話機能のセッションが行う）
2. モデルを書き出し直したら、Hugging Face に新しい commit として上げ、`defaultRevision` を更新して example で確かめる
3. pub.dev の publisher を付けるか決める（今は uploader のまま。pub.dev のサイトでログインして行う。オーナーの作業）
4. 新しい版を出すときは、名前・版・中身と dry-run をオーナーに見せてから

## 申し送り

- 名前は既存とぶつからないことを確かめた（2026-10-21）: Hugging Face に `kiarina/MeanVC2-ONNX` は無い。似た名前に LavMic の `djihuiyong/lavmic-meanvc2-onnx`（2026-10-18、Android の短い区間の変換、ASR なし、速さ・聴取の主張なし）がある。
  pub.dev・GitHub に `meanvc2_onnx` は無い
- ライセンスの確認結果は `huggingface/NOTICE`。話者の encoder は microsoft/UniSpeech の CC BY-SA 3.0。
- MeanVC2 の GitHub には LICENSE のファイルが無く、README の文と Hugging Face のメタデータが Apache-2.0
