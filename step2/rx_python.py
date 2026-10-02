"""Fallback for Step 2b if DPDK will not build or run: reassemble frames from the PCAP in
Python (scapy). Writes the same frames_rx.bin format as dpdk_rx.c.

Usage: python3 rx_python.py [images.pcap] [frames_rx.bin]
"""
import struct
import sys

from scapy.all import UDP, PcapReader

MAGIC = 0x542D0CA0
HDR = struct.Struct("<IIHHIHHI")

pcap = sys.argv[1] if len(sys.argv) > 1 else "images.pcap"
out_path = sys.argv[2] if len(sys.argv) > 2 else "frames_rx.bin"

frames = {}
pkts = bad = dup = 0
for pkt in PcapReader(pcap):
    pkts += 1
    if UDP not in pkt or pkt[UDP].dport != 5000:
        bad += 1
        continue
    payload = bytes(pkt[UDP].payload)
    magic, fid, seq, nseg, off, length, _, fbytes = HDR.unpack_from(payload)
    if magic != MAGIC:
        bad += 1
        continue
    f = frames.setdefault(fid, {"buf": bytearray(fbytes), "nseg": nseg, "seen": set()})
    if seq in f["seen"]:
        dup += 1
        continue
    f["buf"][off:off + length] = payload[HDR.size:HDR.size + length]
    f["seen"].add(seq)

complete = 0
with open(out_path, "wb") as out:
    for fid in sorted(frames):
        f = frames[fid]
        if len(f["seen"]) == f["nseg"]:
            out.write(struct.pack("<I", fid) + f["buf"])
            complete += 1
        else:
            print(f"frame {fid} incomplete: {len(f['seen'])}/{f['nseg']} segments")
print(f"packets={pkts} dropped={bad} duplicates={dup}")
print(f"frames complete={complete} incomplete={len(frames) - complete} -> {out_path}")
