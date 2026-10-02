"""Step 2a: pack images into UDP packets and save them as a PCAP file.

Each 224x224x3 uint8 image (150,528 bytes) is split into chunks of up to 1400 bytes.
Every UDP payload starts with a 24-byte little-endian header:

    magic u32 | frame_id u32 | seq u16 | nseg u16 | offset u32 | length u16 | pad u16 | frame_bytes u32

Outputs:
    images.pcap   the packet stream (replayed by DPDK or rx_python.py)
    frames.npy    the original images, shape (N, 224, 224, 3), for the correctness check

Usage:
    python3 make_pcap.py                 # 16 synthetic images
    python3 make_pcap.py --images DIR    # JPEG/PNG files from DIR (resized to 224x224)
    python3 make_pcap.py --shuffle       # send packets out of order to test reassembly
"""
import argparse
import random
import struct
from pathlib import Path

import numpy as np
from scapy.all import IP, UDP, Ether, Raw, wrpcap

MAGIC = 0x542D0CA0
HDR = struct.Struct("<IIHHIHHI")  # 24 bytes
CHUNK = 1400
UDP_PORT = 5000
H = W = 224


def load_images(n, folder):
    if folder is None:
        # Smooth synthetic images (gradients plus a few shapes), deterministic.
        rng = np.random.default_rng(542)
        yy, xx = np.mgrid[0:H, 0:W]
        imgs = []
        for i in range(n):
            base = np.stack([(xx * (i + 1)) % 256, (yy * (i + 2)) % 256,
                             ((xx + yy) * (i + 3)) % 256], axis=-1)
            noise = rng.integers(0, 32, size=(H, W, 3))
            imgs.append(((base + noise) % 256).astype(np.uint8))
        return np.stack(imgs)
    from PIL import Image
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not files:
        raise SystemExit(f"no images found in {folder}")
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((W, H))) for p in files])


def packets_for(frame_id, img):
    data = img.tobytes()
    nseg = (len(data) + CHUNK - 1) // CHUNK
    for seq in range(nseg):
        off = seq * CHUNK
        chunk = data[off:off + CHUNK]
        hdr = HDR.pack(MAGIC, frame_id, seq, nseg, off, len(chunk), 0, len(data))
        yield (Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
               / IP(src="10.0.0.1", dst="10.0.0.2")
               / UDP(sport=40000, dport=UDP_PORT)
               / Raw(hdr + chunk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=16, help="number of images (max 32 = engine batch limit)")
    ap.add_argument("--images", help="folder of JPEG/PNG images (default: synthetic)")
    ap.add_argument("--shuffle", action="store_true", help="randomize packet order")
    ap.add_argument("--out", default="images.pcap")
    args = ap.parse_args()

    imgs = load_images(args.n, args.images)
    pkts = [p for i, img in enumerate(imgs) for p in packets_for(i, img)]
    if args.shuffle:
        random.Random(0).shuffle(pkts)
    wrpcap(args.out, pkts)
    np.save("frames.npy", imgs)
    print(f"wrote {args.out}: {len(imgs)} images, {len(pkts)} packets "
          f"({len(pkts) // len(imgs)} per image), shuffle={args.shuffle}")
    print("wrote frames.npy", imgs.shape)


if __name__ == "__main__":
    main()
