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
  version 0x14 (Altiverb 7, block size 20) - 24 byte header:
      s[0], s[1], s[2], 0xF167A67F, N, 0x80000000

  s[0..2] are the first three samples, N the sample count. Every following sample is coded as
  the residual r[n] = (3*s[n-1] - 3*s[n-2] + s[n-3]) - s[n], in blocks of <block size> residuals
  (the first block has 3 fewer, the last one covers samples up to N-1):
      6 bits      k = smallest value with 2**k > max|r| in the block
      k+1 bits    each residual, two's complement

Every file is checked while decoding: each block's k must be the minimal one the encoder would
pick, and the stream must end exactly where the file does with only zero padding after it. A file
that passes is bit-for-bit what the encoder would write for the decoded samples, so the WAV is an
exact copy of the original IR.

Altiverb 7 libraries come as one .irbulk file holding the same coded streams (version 0x14,
marker 0xF167A67F, block size 20, levels stored as recorded) plus pictures and a file table; pass
the .irbulk as a source to convert it (see convert_irbulk).

Usage:
    python3 Alti.py [source_folder_or_irbulk ...] [-o output_folder]

Without arguments it converts the three "IR Installer */data/items" folders next to this script
into "Altiverb 6 Library (decoded)" next to them. The folder tree is mirrored (with the leading
'%' that Altiverb uses on folder names removed) and each IR channel becomes <name>.wav. Pictures and
movies go into a "00 DEFAULT PICTURES" folder the way Convology libraries are laid out. Convology
shows a picture whose name matches the start of the IR file name, so inside an IR folder pictures
are named after the common start of its IR names ("Main Hall config b.jpg" serves all the
"Main Hall config b.*.wav"), as .jpg and .png; IR folders without a picture borrow
the venue's, or the closest-named sibling folder's.
info.iri and .kmz files are copied along.

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
from PIL import Image

MARKERS = {0xF167A670: (0x11, 5), 0xF167A675: (0x12, 9), 0xF167A67A: (0x13, 6), 0xF167A67F: (0x14, 6)}
CHANNEL_SUFFIX = re.compile(r"\.(\d|L|R|C|Ls|Rs|LFE|l|r|c|ls|rs)$")
COPY_EXT = {".kmz", ".iri"}
PICTURE_EXT = {".jpg", ".jpeg", ".png", ".gif"}
PICTURES_DIR = "00 DEFAULT PICTURES"


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


def picture_names(wav_names):
    """Convology shows a picture whose name matches the start of the IR name ("Room.png" serves
    "Room Front.wav" and "Room Rear.wav"), so name pictures after the common start of all the IR
    names in the folder. If they share none, every IR gets its own copy."""
    prefix = os.path.commonprefix(wav_names).rstrip(" ._-")
    return [prefix] if prefix else list(wav_names)


def write_pictures(pictures, movies, dst, wav_names, stats, log):
    """Pictures go in "00 DEFAULT PICTURES" as .jpg and .png. In an IR folder the first one is named
    after its IRs (see picture_names), further ones get "_2", "_3"...; elsewhere they keep their
    own names. Movies go in the same folder."""
    if not pictures and not movies:
        return
    out = os.path.join(dst, PICTURES_DIR)
    os.makedirs(out, exist_ok=True)
    for i, src in enumerate(pictures, 1):
        if wav_names:
            names = [n if i == 1 else "%s_%d" % (n, i) for n in picture_names(wav_names)]
        else:
            names = [os.path.splitext(os.path.basename(src))[0]]
        try:
            with Image.open(src) as im:
                for name in names:
                    if im.format == "JPEG":
                        shutil.copyfile(src, os.path.join(out, name + ".jpg"))
                    else:
                        im.convert("RGB").save(os.path.join(out, name + ".jpg"), quality=95)
                    if im.format == "PNG":
                        shutil.copyfile(src, os.path.join(out, name + ".png"))
                    else:
                        im.save(os.path.join(out, name + ".png"))
            stats["pictures"] += 1
        except OSError as e:
            log.write("picture not readable, skipped: %s (%s)\n" % (src, e))
    for src in movies:
        shutil.copyfile(src, os.path.join(out, os.path.basename(src)))


def readable_pictures(paths):
    good = []
    for p in paths:
        try:
            with Image.open(p):
                good.append(p)
        except OSError:
            pass
    return good


def own_pictures(folder, files):
    pictures = []
    for f in sorted(files):
        if f.startswith("."):
            continue
        src = os.path.join(folder, f)
        if os.path.splitext(f)[1].lower() in PICTURE_EXT:
            pictures.append(src)
        elif CHANNEL_SUFFIX.search(f):
            with open(src, "rb") as fh:
                if fh.read(3) == b"\xff\xd8\xff":  # a few channel files are mislabelled JPEGs
                    pictures.append(src)
    return readable_pictures(pictures)


def borrowed_pictures(root, src_root, all_pictures):
    """For an IR folder without a picture: the nearest parent folder's pictures, else those of the
    sibling folder whose name is closest (e.g. "Room m-q" borrows from "Room m-m")."""
    up = os.path.dirname(root)
    while len(up) >= len(src_root):
        if all_pictures.get(up):
            return all_pictures[up]
        up = os.path.dirname(up)
    parent, name = os.path.split(root)
    siblings = [d for d in all_pictures if os.path.dirname(d) == parent and all_pictures[d]]
    if not siblings:
        return []
    best = max(sorted(siblings), key=lambda d: len(os.path.commonprefix([name, os.path.basename(d)])))
    return all_pictures[best]


def convert_tree(src_root, out_root, args, stats, log):
    all_pictures = {root: own_pictures(root, files) for root, dirs, files in os.walk(src_root)}
    for root, dirs, files in os.walk(src_root):
        dirs.sort()
        rel = os.path.relpath(root, src_root)
        dst = out_root if rel == "." else os.path.join(out_root, *[clean(p) for p in rel.split(os.sep)])
        ir_files, movies = [], []
        pictures = all_pictures[root]
        for f in sorted(files):
            if f.startswith(".") or f.startswith("Icon"):
                continue
            src = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if src in pictures or ext in PICTURE_EXT:
                continue
            if CHANNEL_SUFFIX.search(f):
                ir_files.append(f)
            elif ext == ".mov":
                movies.append(src)
            elif ext in COPY_EXT:
                os.makedirs(dst, exist_ok=True)
                shutil.copyfile(src, os.path.join(dst, f))
        if ir_files:
            convert_folder(root, ir_files, dst, args, stats, log)
            if not pictures:
                pictures = borrowed_pictures(root, src_root, all_pictures)
                if pictures:
                    stats["pictures borrowed"] += 1
        write_pictures(pictures, movies, dst, ir_files, stats, log)


def irbulk_table(data, pos):
    """File table of an .irbulk: count, then (name, offset, size) entries."""
    count = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    entries = []
    for _ in range(count):
        n = struct.unpack_from("<I", data, pos)[0]
        name = data[pos + 4:pos + 4 + n].decode("utf-8")
        offset, size = struct.unpack_from("<QQ", data, pos + 4 + n)
        entries.append((name, offset, size))
        pos += 4 + n + 16
    return entries


def convert_irbulk(path, out_root, args, stats, log):
    """Altiverb 7 .irbulk: a database, then pictures and IR channels, then two file tables.
    Header at 0x30 (little-endian u64): 3, 96, ?, picture table offset, IR table offset.
    Picture record: u64 length + JPEG. IR record: u32 1, u32 sample count, u32 3, u32 stream
    length, u32 0, then the same coded stream as Altiverb 6 (version 0x14: block size 20, an
    extra header word 0x80000000 = full scale). Channel levels are stored as recorded, not
    normalised, so no gains are needed."""
    data = open(path, "rb").read()
    if data[:8] != b"_IRBLK3_":
        raise FormatError("not an _IRBLK3_ file")
    pic_table, ir_table = struct.unpack_from("<QQ", data, 0x48)
    pictures = irbulk_table(data, pic_table)
    irs = irbulk_table(data, ir_table)

    # Sample rate: stored as doubles in the database; use it when the bulk has a single one.
    rates = {r for r in (44100.0, 48000.0, 88200.0, 96000.0)
             if struct.pack("<d", r) in data[:min(o for _, o, _ in pictures + irs)]}
    if len(rates) == 1:
        rate = int(rates.pop())
    else:
        rate = 48000
        stats["rate guessed"] += 1
        log.write("no single sample rate found in %s, used 48000\n" % path)

    # Pictures keep their names in their own folder's "00 DEFAULT PICTURES".
    picture_files = {}  # folder -> extracted picture paths
    for name, offset, _ in pictures:
        length = struct.unpack_from("<Q", data, offset)[0]
        folder, base = os.path.split(name)
        out = os.path.join(out_root, folder, PICTURES_DIR)
        os.makedirs(out, exist_ok=True)
        dst = os.path.join(out, base)
        with open(dst, "wb") as fh:
            fh.write(data[offset + 8:offset + 8 + length])
        picture_files.setdefault(folder, []).append(dst)
    for folder, files in picture_files.items():
        pngs = []
        for f in files:
            try:
                with Image.open(f) as im:
                    im.save(os.path.splitext(f)[0] + ".png")
                pngs.append(f)
                stats["pictures"] += 1
            except OSError as e:
                log.write("picture not readable: %s (%s)\n" % (f, e))
        picture_files[folder] = pngs

    folders = {}
    for name, offset, _ in irs:
        folders.setdefault(os.path.dirname(name), []).append((os.path.basename(name), offset))
    for folder, channels in sorted(folders.items()):
        decoded = []
        for base, offset in channels:
            one, count, three, length, _ = struct.unpack_from("<5I", data, offset)
            try:
                if (one, three) != (1, 3):
                    raise FormatError("unknown record %d/%d" % (one, three))
                samples = decode(data[offset + 20:offset + 20 + length])
                if len(samples) != count:
                    raise FormatError("record says %d samples, stream has %d" % (count, len(samples)))
            except FormatError as e:
                stats["failed"] += 1
                log.write("FAILED %s/%s: %s\n" % (folder, base, e))
                print("FAILED %s/%s: %s" % (folder, base, e))
                continue
            decoded.append((base, samples.astype(np.float64) / 2.0 ** 31))
            stats["decoded"] += 1
        if not decoded:
            continue
        dst = os.path.join(out_root, folder)
        os.makedirs(dst, exist_ok=True)
        peak = max(float(np.abs(x).max()) for _, x in decoded) or 1.0
        scale = 10.0 ** (args.peak / 20.0) / peak
        for base, x in decoded:
            out = os.path.join(dst, os.path.splitext(base)[0] + ".wav")
            if args.pcm24:
                sf.write(out, np.round(x * scale * 8388607).astype(np.int32) << 8, rate, subtype="PCM_24")
            else:
                sf.write(out, (x * scale).astype(np.float32), rate, subtype="FLOAT")
        # Pictures from the nearest folder above that has some, named so Convology finds them.
        up = folder
        while up and up not in picture_files:
            up = os.path.dirname(up)
        if picture_files.get(up):
            write_pictures(picture_files[up], [], dst, [os.path.splitext(b)[0] for b, _ in decoded], stats, log)
            stats["pictures borrowed"] += 1


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
    stats = {k: 0 for k in ("decoded", "wav read", "failed", "rate guessed", "gain missing", "pictures", "pictures borrowed")}
    with open(os.path.join(args.output, "conversion log.txt"), "w") as log:
        for s in sources:
            print("Source:", s)
            if s.lower().endswith(".irbulk"):
                convert_irbulk(s, args.output, args, stats, log)
            else:
                convert_tree(s, args.output, args, stats, log)
        log.write("\n%s\n" % stats)
    print(stats)
    print("Output:", args.output)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
