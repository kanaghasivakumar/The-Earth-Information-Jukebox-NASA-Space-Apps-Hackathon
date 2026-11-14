# SoundFont used by `render/fluid.py`

| | |
|---|---|
| File | `FluidR3_GM.sf2` (General MIDI, 148,398,306 bytes) |
| sha256 | `74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0` |
| Authors | Frank Wen (2000–2002, 2008), Toby Smithe (2008) |
| License | **MIT** (see the license text below) |
| Source | Debian package `fluid-soundfont-gm_3.1-6_all.deb` (http://deb.debian.org/debian/pool/main/f/fluid-soundfont/). Debian's copyright file names the upstream source as `http://www.musescore.org/download/fluid-soundfont.tar.gz` |
| Location | `/projects/e32706/omb8654/assets/FluidR3_GM.sf2`, outside git. The copyright file is stored next to it as `FluidR3_GM.COPYING` |

Fetch it with `python -m jukebox.render.soundfont`, which downloads the .deb, extracts the .sf2 and
checks the sha256. You can override the location with `JUKEBOX_SF2=/path/to/file.sf2` or `JUKEBOX_ASSETS=/dir`.
If the binary or the .sf2 is missing, `fluid.render_notes` falls back to `render/synth.py`.

Renderer: the FluidSynth 2.6.2 binary from the `jukebox` conda env (LGPL-2.1). It runs as a separate
process (`fluidsynth -ni -F out.wav ...`) and is not linked into our code.

Credits slide: *"Instrument sounds: FluidR3_GM SoundFont by Frank Wen, MIT license; rendered with FluidSynth."*

## License text (MIT, from the Debian copyright file)

```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Model weights (for the credits slide too)

The harmony comes from the Anticipatory Music Transformer, `stanford-crfm/music-medium-800k` and
`music-small-800k` (Thickstun, Hall, Donahue & Liang, 2023). The weights are Apache-2.0 according to
their Hugging Face model cards. They are cached under `$HF_HOME=/projects/e32706/omb8654/hf_cache`.
