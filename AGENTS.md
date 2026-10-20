# AGENTS.md

このリポジトリで作業するエージェント向けのガイドラインです。

## 作業前に読むもの

- `README.md`（公開向けの説明。使い方・既定・速さ）
- `lib/meanvc2_onnx.dart` と `lib/src/`（Dart の契約。ここが正典）
- `huggingface/README.md`・`huggingface/NOTICE`（Hugging Face のモデルカードと出どころ。ライセンスの正典）
- `tasks/`（残っている作業。1 タスク 1 ファイル）

実証の来歴は labs の `2026/10/21/meanvc2-onnx-streaming`（書き出し・上流との一致・4 プラットフォームの速さ・聴取）。この repo は、その lab の
`py/` の書き出しと `app/lib/meanvc2.dart` を写して作り（2026-10-21）、以後はここが所有する。lab とは同期しない。

## 設計の決まり

- 流しながらの変換は上流 `runtime/run_rt.py` の `VCRunner.process_chunk` の写し。キャッシュ・ASR の窓・BN の補間・ODE・Vocos のつなぎは呼び出す側
  （`lib/src/stream.dart` と `tool/onnx_stream.py`）に置き、グラフは 1 回の呼び出しだけを持つ。**Dart と Python の写しは同じ変更で直す**
- 既定は int8（ASR・DiT・Vocos。fbank・GTM は fp32）。fp32 は一致の基準と切り替え先
- int8 は ONNX Runtime の版で結果が動く。flutter_onnxruntime（とその ORT）の版を上げたら、example の自己確認（`FIXTURE_DIR`）を流し、聞き直す
- 話者の encoder（`speaker/`）は CC BY-SA 3.0。変換のグラフ（Apache-2.0）と同じ場所に混ぜず、ライセンスの表記を分ける（下の「ライセンス」）
- 書き出しは決定的（同じ版の torch・onnxruntime なら同じバイト）。版を上げて書き出し直したら、sha256 が変わるので Hugging Face の新しい commit として出す

## ライセンス

- 変換のグラフは、上流（MeanVC2 の Hugging Face の重み）に従って Apache-2.0
- 話者の encoder は microsoft/UniSpeech の CC BY-SA 3.0。変換のグラフと分けて表記する

## 検証

- `flutter analyze`・`flutter test`（Dart の純粋な部分）
- `tool/export_onnx.py --int8`（部品ごとに上流と比べ、`results/export-check.json`）、`tool/export_speaker.py --check-wav <話し声>`
- example を `MODEL_DIR=build/hf`・`FIXTURE_DIR=build/fixtures` で動かし、`MEANVC2_EXAMPLE` の行の `snrDbVsPython`（fp32 は 50〜60 dB）と `speakerCosineVsUpstream`（fp32 は 1.0）を見る

## 公開

Hugging Face のアップロード・公開、GitHub のリポジトリを公開にすること、pub.dev への publish は、どれも取り消しにくい。
**名前・版・ファイルの一覧と大きさ・モデルカードと README の本文・ライセンスの確認結果をオーナーに見せ、承認を得てから行う。** 進み具合は `tasks/publish.md`。

## 決まりの出どころ

| 決まり | 決めた人 | 理由 |
|---|---|---|
| 話者の encoder は CC BY-SA 3.0 として分ける | エージェント、2026-10-21 | 元の重み（microsoft/UniSpeech）のライセンス |
| 既定は int8、fp32 は基準と切り替え先 | エージェント、2026-10-21 | オーナーの聴取で差がほとんど分からず、56 MB で速い（labs の lab） |
