import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:meanvc2_onnx/meanvc2_onnx.dart';

void main() {
  test('interpolation matches align_corners=True', () {
    // 5 rows of 1 channel -> 17 rows, as upstream does for one BN chunk.
    final x = Float32List.fromList([0, 1, 2, 3, 4]);
    final y = interpLinearAlignCorners(x, 5, 17, 1);
    expect(y.length, 17);
    for (var j = 0; j < 17; j++) {
      expect(y[j], closeTo(j * 4 / 16, 1e-6));
    }
    expect(interpLinearAlignCorners(x, 5, 17, 1, dropFirst: true).length, 16);
  });

  test('crossfade joins chunks like upstream', () {
    final first = crossfadeJoin(null, Float32List.fromList(List.filled(8, 1)), 4);
    expect(first, everyElement(1));
    expect(first.length, 4);
    final next = crossfadeJoin(Float32List.fromList([1, 1, 1, 1]), Float32List.fromList(List.filled(8, 0)), 4);
    expect(next.toList(), [1, closeTo(2 / 3, 1e-6), closeTo(1 / 3, 1e-6), 0]);
  });

  test('file names per variant', () {
    expect(MeanVc2Models.files(ModelVariant.int8),
        ['fbank.onnx', 'asr.int8.onnx', 'gtm.onnx', 'dit.int8.onnx', 'vocos.int8.onnx']);
    expect(MeanVc2Models.files(ModelVariant.fp32, speaker: ModelVariant.fp32).last, 'speaker/speaker.onnx');
  });
}
