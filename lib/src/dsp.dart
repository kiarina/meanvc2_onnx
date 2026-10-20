import 'dart:math' as math;
import 'dart:typed_data';

/// `F.interpolate(mode='linear', align_corners=True)` along the rows of a
/// row-major [n] x [c] matrix, to [size] rows (the first dropped when
/// [dropFirst], as upstream does after upsampling the BN features).
Float32List interpLinearAlignCorners(Float32List x, int n, int size, int c, {bool dropFirst = false}) {
  final start = dropFirst ? 1 : 0;
  final out = Float32List((size - start) * c);
  if (n == 1) {
    for (var j = 0; j < size - start; j++) {
      out.setRange(j * c, (j + 1) * c, x);
    }
    return out;
  }
  final scale = (n - 1) / (size - 1);
  for (var j = start; j < size; j++) {
    final pos = j * scale;
    final lo = math.min(pos.floor(), n - 2);
    final w = pos - lo;
    final o = (j - start) * c;
    for (var k = 0; k < c; k++) {
      out[o + k] = x[lo * c + k] * (1 - w) + x[(lo + 1) * c + k] * w;
    }
  }
  return out;
}

/// Upstream's vocoder overlap: the first [overlap] samples of [wav] are
/// cross-faded with [last] (the tail kept from the previous chunk, or none for
/// the first chunk), and the last [overlap] samples are held back.
Float32List crossfadeJoin(Float32List? last, Float32List wav, int overlap) {
  if (last == null) return Float32List.fromList(wav.sublist(0, wav.length - overlap));
  final out = Float32List(wav.length - overlap);
  for (var k = 0; k < overlap; k++) {
    final up = k / (overlap - 1);
    out[k] = last[k] * (1 - up) + wav[k] * up;
  }
  out.setAll(overlap, wav.sublist(overlap, wav.length - overlap));
  return out;
}
