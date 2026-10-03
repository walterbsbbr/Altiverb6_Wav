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

Options: `--pcm24` (24-bit PCM output), `--raw` (no info.iri gains), `--peak <dBFS>` (level of the
loudest channel per IR folder, default -0.1).
