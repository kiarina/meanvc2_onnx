import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// Makes the 256-d speaker embedding of a reference voice (WavLM-Large +
/// ECAPA-TDNN, as upstream `preprocess/models/ecapa_tdnn.py`). Run it once
/// when the reference is registered and keep the embedding; it is not needed
/// during conversion. The fp32 graph is 1.3 GB, the int8 one 387 MB.
class SpeakerEncoder {
  SpeakerEncoder._(this._session);

  final OrtSession _session;

  static Future<SpeakerEncoder> load(String path, {int? threads}) async {
    final session = await OnnxRuntime().createSession(path,
        options: OrtSessionOptions(intraOpNumThreads: threads, providers: const [OrtProvider.CPU]));
    return SpeakerEncoder._(session);
  }

  /// [wav]: the reference, 16 kHz mono float (a few seconds of clean speech).
  Future<Float32List> embed(Float32List wav) async {
    final input = await OrtValue.fromList(wav, [1, wav.length]);
    try {
      final out = await _session.run({'wav': input});
      final v = out['embedding']!;
      final flat = await v.asFlattenedList();
      await v.dispose();
      return flat is Float32List ? flat : Float32List.fromList([for (final x in flat) (x as num).toDouble()]);
    } finally {
      await input.dispose();
    }
  }

  Future<void> close() => _session.close();
}
