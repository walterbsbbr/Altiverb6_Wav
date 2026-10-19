# Altiverb6_Wav

Lossless converter for Altiverb 6 impulse-response libraries (IR Installer CDs) and Altiverb 7
`.irbulk` files to plain WAV files for generic convolution apps such as Convology.

Most Altiverb 6 channel files (`.1 .2 .3 .4 .L .R .C .Ls .Rs`) are not raw PCM but Audio Ease's own
compressed format (`cir2` and two older variants): a fixed 3rd-order predictor with residuals packed
in small blocks. `Alti.py` decodes them exactly (each file is checked against the encoder's rules),
applies the per-channel gains and sample rates from `info.iri`, and writes 32-bit float WAVs in a
mirrored folder tree. See the docstring at the top of `Alti.py` for the format details.

    pip install numpy soundfile
    python3 Alti.py -o "<output folder>"          # converts the IR Installer folders next to this repo
    python3 Alti.py <items folder> -o "<output>"   # or any folder of Altiverb IRs
    python3 Alti.py <file.irbulk> -o "<output>"    # an Altiverb 7 IR bulk file

Options: `--pcm24` (24-bit PCM output), `--raw` (no info.iri gains), `--join` (one multichannel
file per IR instead of one mono file per channel), `--peak <dBFS>` (level of the loudest channel
per IR folder, default -0.1).

By default each IR channel becomes its own mono `<name>.wav`. With `--join` each IR becomes one
file: a stereo pair becomes a stereo file, a true stereo set becomes one four channel file in
LL, LR, RL, RR order (the layout a convolution plug-in expects for quad true stereo), and a
surround set becomes one four channel file in the standard L, R, Ls, Rs order. Channels shorter
than the longest are padded with silence. IRs whose channel order cannot be derived from their
names, the eight and sixteen channel surround sets in particular, are left as separate mono files
and listed in the log. The order comes from the channel names, which were checked against the
libraries: the same audio is stored as `...1` in one folder and `...L` in another, so the numbers
are the standard ones (1 = L, 2 = R, 3 = Ls, 4 = Rs).

## Interface (PyQt5)

    pip install PyQt5 numpy soundfile pillow
    python3 AltiGUI.py            # or double-click AltiGUI.command

Choose a source folder (Altiverb 6 IR folders and/or Altiverb 7 `.irbulk` files anywhere inside
it) and an output folder, then Convert. Same conversion and options as `Alti.py`, with a progress
bar, a Stop button and the log in the window.
