// Example and self-check for meanvc2_onnx.
//
// Downloads the models (or reads MODEL_DIR), registers the reference voice
// with the speaker encoder, converts the source in 160 ms chunks and, when the
// check fixtures are given (FIXTURE_DIR, made by tool/make_check_fixture.py),
// compares the result with the Python ONNX Runtime output for the same noise.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:meanvc2_onnx/meanvc2_onnx.dart';
import 'package:path_provider/path_provider.dart';

const modelDir = String.fromEnvironment('MODEL_DIR');
const fixtureDir = String.fromEnvironment('FIXTURE_DIR');
const revision = String.fromEnvironment('REVISION', defaultValue: 'main');
const variantName = String.fromEnvironment('VARIANT', defaultValue: 'int8');
const threads = int.fromEnvironment('ORT_THREADS', defaultValue: 2);
const autoExit = bool.fromEnvironment('AUTO_EXIT');

void main() => runApp(const ExampleApp());

class ExampleApp extends StatefulWidget {
  const ExampleApp({super.key});
  @override
  State<ExampleApp> createState() => _ExampleAppState();
}

class _ExampleAppState extends State<ExampleApp> {
  final _lines = <String>[];

  void _log(String s) {
    debugPrint('[meanvc2_onnx] $s');
    setState(() => _lines.add(s));
  }

  @override
  void initState() {
    super.initState();
    _run();
  }

  Float32List _f32(String path) {
    final b = File(path).readAsBytesSync();
    return b.buffer.asFloat32List(b.offsetInBytes, b.lengthInBytes ~/ 4);
  }

  Future<void> _run() async {
    final variant = ModelVariant.values.byName(variantName);
    final report = <String, dynamic>{'platform': Platform.operatingSystem, 'variant': variantName, 'threads': threads};
    try {
      var dir = modelDir;
      if (dir.isEmpty) {
        dir = '${(await getApplicationSupportDirectory()).path}/meanvc2_onnx';
        await MeanVc2Models.download(dir,
            revision: revision, files: MeanVc2Models.files(variant, speaker: variant), onFile: (f) => _log('download $f'));
      }
      // 1. register the reference voice (once per voice; keep the embedding)
      Float32List? spk;
      if (fixtureDir.isNotEmpty && File('$fixtureDir/target.f32').existsSync()) {
        final encoder = await SpeakerEncoder.load('$dir/${MeanVc2Models.speakerFileName(variant)}', threads: threads);
        final sw = Stopwatch()..start();
        spk = await encoder.embed(_f32('$fixtureDir/target.f32'));
        report['speakerMs'] = sw.elapsedMilliseconds;
        await encoder.close();
        final ref = _f32('$fixtureDir/spk.f32');
        report['speakerCosineVsUpstream'] = _cosine(spk, ref);
        _log('speaker ${report['speakerMs']} ms, cosine vs upstream ${report['speakerCosineVsUpstream']}');
        spk = ref; // the check below uses the fixture embedding so that it compares the conversion only
      }
      // 2. convert in 160 ms chunks
      final vc = await MeanVc2Stream.load(dir, variant: variant, threads: threads, steps: 3);
      await vc.setTarget(spk ?? _f32('$fixtureDir/spk.f32'));
      final source = _f32('$fixtureDir/source.f32');
      if (File('$fixtureDir/noise.f32').existsSync()) {
        final noise = _f32('$fixtureDir/noise.f32');
        var pos = 0;
        vc.noiseFn = (n) => Float32List.fromList(Float32List.sublistView(noise, pos, pos += n * 80));
      }
      final out = <double>[];
      final times = <double>[];
      for (var i = 0; i < source.length; i += kChunk) {
        final chunk = Float32List(kChunk)..setAll(0, Float32List.sublistView(source, i, math.min(i + kChunk, source.length)));
        final sw = Stopwatch()..start();
        out.addAll(await vc.process(chunk));
        times.add(sw.elapsedMicroseconds / 1000);
      }
      await vc.close();
      times.sort();
      report['chunkMsP50'] = times[times.length ~/ 2];
      final expected = File('$fixtureDir/expected_${variantName}_s3.f32');
      if (expected.existsSync()) {
        final e = _f32(expected.path);
        double err = 0, ref = 0;
        for (var i = 0; i < math.min(e.length, out.length); i++) {
          err += (out[i] - e[i]) * (out[i] - e[i]);
          ref += e[i] * e[i];
        }
        report['snrDbVsPython'] = 10 * math.log(ref / (err + 1e-30)) / math.ln10;
        report['samples'] = [out.length, e.length];
      }
      _log(jsonEncode(report));
    } catch (e, st) {
      _log('error: $e\n$st');
    }
    debugPrint('MEANVC2_EXAMPLE ${jsonEncode(report)}');
    if (autoExit) exit(0);
  }

  static double _cosine(Float32List a, Float32List b) {
    double ab = 0, aa = 0, bb = 0;
    for (var i = 0; i < a.length; i++) {
      ab += a[i] * b[i];
      aa += a[i] * a[i];
      bb += b[i] * b[i];
    }
    return ab / math.sqrt(aa * bb);
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('meanvc2_onnx')),
          body: ListView(padding: const EdgeInsets.all(12), children: [for (final l in _lines) SelectableText(l)]),
        ),
      );
}
