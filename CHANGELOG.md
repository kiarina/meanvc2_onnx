## 0.1.0

- First release on pub.dev. Same API as 0.1.0-dev.1, which was used as a git
  dependency in a real-time call app on macOS and Windows.
- `MeanVc2Models.defaultRevision` is the Hugging Face commit
  2db97430c6e315a819daa30e255e14615205a3d1.
- README: how to play the output in real time (playback headroom, counting
  gaps only while there is voice).

## 0.1.0-dev.1

- First version: `MeanVc2Stream` (160 ms streaming conversion, fp32 or int8),
  `SpeakerEncoder` (256-d embedding of a reference voice), `MeanVc2Models`
  (sha256-checked download from the Hugging Face repository, pinned to
  commit 2db97430c6e315a819daa30e255e14615205a3d1 by default).
