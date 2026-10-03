"""
Lossless converter for Altiverb 6 impulse-response files (.1 .2 .3 .4 .L .R .C .Ls .Rs ...) to WAV.

Altiverb stores each IR channel in Audio Ease's own compressed format (read from the encoder in
the IR Installer that ships with the library). It is NOT raw PCM, which is why reading the bytes
directly only produced noise for most files.

Format, all big-endian, written MSB-first as one bit stream:

  version 0x12 ("cir2", block size 18) - 36 byte header:
      'cir2', 0, 0, 0xF167A675, 0, s[0], s[1], s[2], N
  version 0x11 (no magic, block size 17) - 20 byte header:
      s[0], s[1], s[2], 0xF167A670, N
  version 0x13 (block size 19) - 24 byte header:
      s[0], s[1], s[2], 0xF167A67A, N, extra

  s[0..2] are the first three samples, N the sample count. Every following sample is coded as
  the residual r[n] = (3*s[n-1] - 3*s[n-2] + s[n-3]) - s[n], in blocks of <block size> residuals
  (the first block has 3 fewer, the last one covers samples up to N-1):
      6 bits      k = smallest value with 2**k > max|r| in the block
      k+1 bits    each residual, two's complement

Every file is checked while decoding: each block's k must be the minimal one the encoder would
pick, and the stream must end exactly where the file does with only zero padding after it. A file
that passes is bit-for-bit what the encoder would write for the decoded samples, so the WAV is an
exact copy of the original IR.

Usage:
    python3 Alti.py [source_folder ...] [-o output_folder]

Without arguments it converts the three "IR Installer */data/items" folders next to this script
into "Altiverb 6 Library (decoded)" next to them. The folder tree is mirrored (with the leading
'%' that Altiverb uses on folder names removed), each IR channel becomes <name>.wav, and pictures,
movies and info.iri files are copied along.

Levels: Audio Ease normalised every channel file to full scale and stored the gain that restores
it as the 4th field of its info.iri line ("name: 3 <rate> <samples> <gain dB> ..."). The converter
applies those gains (checked on the library: it brings the late-tail levels of the channels in a
folder together, the opposite sign drives them apart), then scales each folder as a whole so its
loudest channel peaks at -0.1 dBFS. Balance between L/R/surround channels is kept that way.
Sample rate (48 or 44.1 kHz) also comes from info.iri. Output is 32-bit float WAV, or 24-bit
PCM with --pcm24; --raw writes the stored samples without the gains.
"""

import argparse
import os
import re
import shutil
import struct
import sys

import numpy as np
import soundfile as sf

MARKERS = {0xF167A670: (0x11, 5), 0xF167A675: (0x12, 9), 0xF167A67A: (0x13, 6)}
CHANNEL_SUFFIX = re.compile(r"\.(\d|L|R|C|Ls|Rs|LFE|l|r|c|ls|rs)$")
COPY_EXT = {".jpg", ".jpeg", ".png", ".gif", ".mov", ".kmz", ".iri"}


class FormatError(Exception):
    pass


def parse_header(data):
    if len(data) < 24:
        raise FormatError("file too short")
    words = struct.unpack(">6I", data[:24])
    if data[:4] == b"cir2":
        words = struct.unpack(">9i", data[:36])
        if (words[3] & 0xFFFFFFFF) != 0xF167A675:
            raise FormatError("cir2 without 0xF167A675 marker")
        return 18, 36, list(words[5:8]), words[8]
    marker = words[3]
    if marker not in MARKERS:
        raise FormatError("unknown header")
    version, nwords = MARKERS[marker]
    signed = struct.unpack(">%di" % nwords, data[: nwords * 4])
    return version, nwords * 4, list(signed[0:3]), signed[4]


def decode(data):
    """Decode one Altiverb IR channel. Returns int64 samples; raises FormatError if not exact."""
    block, start, warm, n = parse_header(data)
    if n < 3:
        raise FormatError("sample count %d" % n)

    buf = np.frombuffer(data + bytes(16), dtype=np.uint8)
    total_bits = len(data) * 8

    def read(pos, nbits):  # scalar, used for the 6-bit block headers
        b = pos >> 3
        w = int.from_bytes(data[b:b + 8].ljust(8, b"\0"), "big")
        return (w >> (64 - (pos & 7) - nbits)) & ((1 << nbits) - 1)

    # Walk the block headers; residual positions follow from each k.
    pos = start * 8
    counts, widths, starts = [], [], []
    off, first = 0, True
    while off < n:
        lim = min(block, n - off - 1)
        cnt = lim - 2 if first else lim
        first = False
        if pos + 6 > total_bits:
            raise FormatError("stream ends early")
        k = read(pos, 6)
        pos += 6
        if k > 31:
            raise FormatError("residual width %d" % k)
        counts.append(cnt)
        widths.append(k + 1)
        starts.append(pos)
        pos += cnt * (k + 1)
        off += block
    end = pos
    if end > total_bits:
        raise FormatError("stream runs past end of file")

    counts = np.array(counts, dtype=np.int64)
    widths = np.array(widths, dtype=np.int64)
    starts = np.array(starts, dtype=np.int64)
    w_each = np.repeat(widths, counts)
    idx_in_block = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    bitpos = np.repeat(starts, counts) + idx_in_block * w_each

    # Gather 8 bytes at each residual's byte position and cut its bits out.
    byte = bitpos >> 3
    word = np.zeros(len(bitpos), dtype=np.uint64)
    for i in range(8):
        word = (word << np.uint64(8)) | buf[byte + i].astype(np.uint64)
    shift = (64 - (bitpos & 7) - w_each).astype(np.uint64)
    raw = ((word >> shift) & ((np.uint64(1) << w_each.astype(np.uint64)) - np.uint64(1))).astype(np.int64)
    res = np.where(raw >= (np.int64(1) << (w_each - 1)), raw - (np.int64(1) << w_each), raw)

    # The encoder always picks the smallest k with 2**k > max|r|: check every block.
    if len(res):
        blk = np.repeat(np.arange(len(counts)), counts)
        maxabs = np.zeros(len(counts), dtype=np.int64)
        np.maximum.at(maxabs, blk, np.abs(res))
        k = widths - 1
        need = np.zeros_like(k)
        nz = maxabs > 0
        need[nz] = np.floor(np.log2(maxabs[nz])).astype(np.int64) + 1
        # log2 can round at exact powers of two; settle with integer comparisons
        need = np.where((np.int64(1) << need) <= maxabs, need + 1, need)
        need = np.where((need > 0) & ((np.int64(1) << (need - 1)) > maxabs), need - 1, need)
        if np.any(need != k):
            raise FormatError("block width not minimal (stream misread)")
    elif np.any(widths != 1):
        raise FormatError("block width not minimal (stream misread)")

    # The file holds exactly the bytes the stream needs, the last one zero-padded.
    if len(data) != (end + 7) // 8:
        raise FormatError("file size %d, stream needs %d" % (len(data), (end + 7) // 8))
    first_pad = data[end >> 3] & (0xFF >> (end & 7)) if end < total_bits else 0
    if first_pad or any(data[(end >> 3) + 1:]):
        raise FormatError("non-zero padding after stream")

    if len(res) != n - 3:
        raise FormatError("residual count %d for %d samples" % (len(res), n))

    # s[n] - 3s[n-1] + 3s[n-2] - s[n-3] = -r[n]: undo the third difference with three cumsums.
    e = np.empty(n, dtype=np.int64)
    s0, s1, s2 = warm
    e[0] = s0
    e[1] = s1 - 3 * s0
    e[2] = s2 - 3 * s1 + 3 * s0
    e[3:] = -res
    out = np.cumsum(np.cumsum(np.cumsum(e)))
    if out[0] != s0 or out[1] != s1 or out[2] != s2:
        raise FormatError("warm-up mismatch")
    return out


def read_info(folder):
    """Per channel file (lower-cased name, info.iri case differs at times): (rate, count, gain dB)."""
    info = {}
    # usually info.iri, in a few folders "<name>.iri"
    for name in sorted(n for n in os.listdir(folder) if n.lower().endswith(".iri")):
        text = open(os.path.join(folder, name), "rb").read().decode("latin-1")
        for line in text.splitlines():
            m = re.match(r"(.+?): \S+ (\d+(?:\.\d+)?) (\d+) (\S+)", line)
            if m:
                # "none": the file was stored at its recorded level, no gain to undo
                try:
                    gain = 0.0 if m.group(4) == "none" else float(m.group(4))
                except ValueError:
                    gain = None
                info[m.group(1).lower()] = (int(round(float(m.group(2)))), int(m.group(3)), gain)
    return info


def clean(part):
    return part[1:] if part.startswith("%") else part


def convert_folder(root, files, dst, args, stats, log):
    info = read_info(root)
    channels = []  # (name, samples as float, rate)
    for f in files:
        src = os.path.join(root, f)
        data = open(src, "rb").read()
        if data[:3] == b"\xff\xd8\xff":
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, f + ".jpg"))  # mislabelled picture
            continue
        rate, count, gain = info.get(f.lower(), (None, None, None))
        try:
            if data[:4] == b"RIFF":
                ints, wav_rate = sf.read(src, dtype="int32")
                samples = (ints >> 8).astype(np.int64) if sf.info(src).subtype == "PCM_24" else ints.astype(np.int64)
                rate = rate or wav_rate
                stats["wav read"] += 1
            else:
                samples = decode(data)
                stats["decoded"] += 1
        except (FormatError, RuntimeError, ValueError) as e:
            stats["failed"] += 1
            log.write("FAILED %s: %s\n" % (src, e))
            print("FAILED %s: %s" % (src, e))
            continue
        if rate is None:
            rate = 48000
            stats["rate guessed"] += 1
            log.write("no rate in info.iri, used 48000: %s\n" % src)
        if count is not None and count != len(samples):
            log.write("info.iri says %d samples, file has %d: %s\n" % (count, len(samples), src))
        if gain is None:
            gain = 0.0
            if not args.raw:
                stats["gain missing"] += 1
                log.write("no gain in info.iri, used 0 dB: %s\n" % src)
        x = samples.astype(np.float64) / (1 << 23)
        if not args.raw:
            x *= 10.0 ** (gain / 20.0)
        channels.append((f, x, rate))

    if not channels:
        return
    os.makedirs(dst, exist_ok=True)
    # One scale for the whole folder keeps the level balance between its channels.
    peak = max(float(np.abs(x).max()) for _, x, _ in channels) or 1.0
    scale = 10.0 ** (args.peak / 20.0) / peak
    for f, x, rate in channels:
        out = os.path.join(dst, f + ".wav")
        if args.pcm24:
            sf.write(out, np.round(x * scale * 8388607).astype(np.int32) << 8, rate, subtype="PCM_24")
        else:
            sf.write(out, (x * scale).astype(np.float32), rate, subtype="FLOAT")
    if stats["decoded"] // 250 != (stats["decoded"] - len(channels)) // 250:
        print("converted %d ..." % (stats["decoded"] + stats["wav read"]), flush=True)


def convert_tree(src_root, out_root, args, stats, log):
    for root, dirs, files in os.walk(src_root):
        dirs.sort()
        rel = os.path.relpath(root, src_root)
        dst = out_root if rel == "." else os.path.join(out_root, *[clean(p) for p in rel.split(os.sep)])
        ir_files = []
        for f in sorted(files):
            if f.startswith(".") or f.startswith("Icon"):
                continue
            if CHANNEL_SUFFIX.search(f):
                ir_files.append(f)
            elif os.path.splitext(f)[1].lower() in COPY_EXT:
                os.makedirs(dst, exist_ok=True)
                shutil.copyfile(os.path.join(root, f), os.path.join(dst, f))
        if ir_files:
            convert_folder(root, ir_files, dst, args, stats, log)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*")
    ap.add_argument("-o", "--output", default=os.path.join(base, "Altiverb 6 Library (decoded)"))
    ap.add_argument("--raw", action="store_true",
                    help="do not apply the per-channel gains from info.iri")
    ap.add_argument("--pcm24", action="store_true", help="write 24-bit PCM instead of 32-bit float")
    ap.add_argument("--peak", type=float, default=-0.1,
                    help="peak level in dBFS of the loudest channel of each IR folder (default -0.1)")
    args = ap.parse_args()
    sources = args.sources or [
        os.path.join(base, d, "data", "items")
        for d in ("IR Installer Complete", "IR Installer Omnisound", "IR Installer Zuylen2")
        if os.path.isdir(os.path.join(base, d, "data", "items"))
    ]
    os.makedirs(args.output, exist_ok=True)
    stats = {k: 0 for k in ("decoded", "wav read", "failed", "rate guessed", "gain missing")}
    with open(os.path.join(args.output, "conversion log.txt"), "w") as log:
        for s in sources:
            print("Source:", s)
            convert_tree(s, args.output, args, stats, log)
        log.write("\n%s\n" % stats)
    print(stats)
    print("Output:", args.output)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
