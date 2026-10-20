import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

/// Weights of the conversion graphs. `int8` quantizes the MatMul/Gemm weights
/// of `asr`, `dit` and `vocos`; `fbank` and `gtm` are always fp32.
enum ModelVariant { fp32, int8 }

/// The graphs [MeanVc2Stream] loads.
const conversionGraphs = ['fbank', 'asr', 'gtm', 'dit', 'vocos'];

/// File names and downloads for the Hugging Face repository.
class MeanVc2Models {
  static const repo = 'kiarina/MeanVC2-ONNX';
  static const _int8 = {'asr', 'dit', 'vocos'};

  static String fileName(String graph, ModelVariant variant) =>
      variant == ModelVariant.int8 && _int8.contains(graph) ? '$graph.int8.onnx' : '$graph.onnx';

  /// The speaker encoder (WavLM-Large + ECAPA-TDNN). Licensed CC BY-SA 3.0,
  /// unlike the conversion graphs (see NOTICE).
  static String speakerFileName(ModelVariant variant) =>
      variant == ModelVariant.int8 ? 'speaker/speaker.int8.onnx' : 'speaker/speaker.onnx';

  /// Files to fetch for [variant]; [speaker] adds the speaker encoder.
  static List<String> files(ModelVariant variant, {ModelVariant? speaker}) => [
        for (final g in conversionGraphs) fileName(g, variant),
        if (speaker != null) speakerFileName(speaker),
      ];

  /// Downloads [files] of [revision] (pin a commit hash) into [dir], checking
  /// each against the sha256 in the repository's `manifest.json`. Files that
  /// are already there with the right hash are kept.
  static Future<void> download(String dir, {required String revision, required List<String> files,
      void Function(String file)? onFile, http.Client? client}) async {
    final c = client ?? http.Client();
    try {
      final base = 'https://huggingface.co/$repo/resolve/$revision';
      final res = await c.get(Uri.parse('$base/manifest.json'));
      if (res.statusCode != 200) throw HttpException('manifest.json: ${res.statusCode}');
      final manifest = (jsonDecode(res.body) as Map<String, dynamic>)['files'] as Map<String, dynamic>;
      for (final name in files) {
        final entry = manifest[name] as Map<String, dynamic>?;
        if (entry == null) throw ArgumentError('not in manifest: $name');
        final file = File('$dir/$name');
        if (await file.exists() && await _sha256(file) == entry['sha256']) continue;
        onFile?.call(name);
        await file.parent.create(recursive: true);
        final part = File('${file.path}.part');
        final req = http.Request('GET', Uri.parse('$base/$name'));
        final stream = await c.send(req);
        if (stream.statusCode != 200) throw HttpException('$name: ${stream.statusCode}');
        await stream.stream.pipe(part.openWrite());
        if (await _sha256(part) != entry['sha256']) {
          await part.delete();
          throw StateError('sha256 mismatch: $name');
        }
        await part.rename(file.path);
      }
    } finally {
      if (client == null) c.close();
    }
  }

  static Future<String> _sha256(File f) async => (await sha256.bind(f.openRead()).first).toString();
}
