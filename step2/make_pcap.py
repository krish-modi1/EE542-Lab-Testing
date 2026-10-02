"""Split 16 test images into UDP packets and save them to images.pcap.

Each 224x224x3 image (150,528 bytes) becomes 108 packets of up to 1400 bytes.
Every UDP payload is a 24-byte header followed by one chunk of the image.
Header fields (little-endian):
    magic u32 | frame_id u32 | seq u16 | nseg u16 | offset u32 | length u16 | pad u16 | frame_bytes u32
The original images are saved to frames.npy so the receiver's output can be checked.

Usage: python3 make_pcap.py            # packets in order
       python3 make_pcap.py --shuffle  # packets out of order, to test reassembly
"""
import random
import struct
import sys

import numpy as np
from scapy.all import IP, UDP, Ether, Raw, wrpcap

N_IMAGES, CHUNK, PORT, MAGIC = 16, 1400, 5000, 0x542D0CA0
HEADER = struct.Struct("<IIHHIHHI")  # 24 bytes, must match struct frag_hdr in dpdk_rx.c

# Synthetic test images: colour gradients plus a little noise, the same on every run.
rng = np.random.default_rng(0)
y, x = np.mgrid[0:224, 0:224]
images = np.stack([
    (np.stack([x * (i + 1), y * (i + 2), (x + y) * (i + 3)], axis=-1)
     + rng.integers(0, 32, (224, 224, 3))) % 256
    for i in range(N_IMAGES)
]).astype(np.uint8)

packets = []
for frame_id, image in enumerate(images):
    data = image.tobytes()
    nseg = (len(data) + CHUNK - 1) // CHUNK
    for seq in range(nseg):
        chunk = data[seq * CHUNK:(seq + 1) * CHUNK]
        header = HEADER.pack(MAGIC, frame_id, seq, nseg, seq * CHUNK, len(chunk), 0, len(data))
        packets.append(Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
                       / IP(src="10.0.0.1", dst="10.0.0.2")
                       / UDP(sport=40000, dport=PORT)
                       / Raw(header + chunk))

if "--shuffle" in sys.argv:
    random.Random(0).shuffle(packets)

wrpcap("images.pcap", packets)
np.save("frames.npy", images)
print(f"wrote images.pcap ({len(packets)} packets) and frames.npy {images.shape}")
