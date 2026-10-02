/*
 * Step 2b: DPDK receiver. Reads packets from a PCAP file through DPDK's pcap poll-mode
 * driver (no physical NIC needed), parses Ethernet/IPv4/UDP headers, and reassembles the
 * image frames written by make_pcap.py.
 *
 * With DOCA GPUNetIO, a CUDA kernel would do this parsing and the NIC would write the
 * payload straight into GPU memory. Here the CPU does it, and the frames are written to
 * frames_rx.bin for infer_rx.py to move into GPU memory.
 *
 * Run (no hugepages, no PCI devices):
 *   ./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
 *       --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null' -- frames_rx.bin
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_udp.h>

#define MAGIC 0x542D0CA0u
#define UDP_PORT 5000
#define MAX_FRAMES 64
#define MAX_SEGS 256
#define BURST 32
#define IDLE_POLLS 200000 /* stop after this many empty polls in a row */

struct __attribute__((packed)) frag_hdr {
	uint32_t magic, frame_id;
	uint16_t seq, nseg;
	uint32_t offset;
	uint16_t length, pad;
	uint32_t frame_bytes;
};

struct frame {
	uint8_t *data;
	uint32_t bytes, received, nseg, got;
	uint8_t seen[MAX_SEGS];
};

static struct frame frames[MAX_FRAMES];

int main(int argc, char **argv)
{
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "EAL init failed\n");
	argc -= ret;
	argv += ret;
	const char *out_path = argc > 1 ? argv[1] : "frames_rx.bin";

	if (rte_eth_dev_count_avail() == 0)
		rte_exit(EXIT_FAILURE, "no ports: did you pass --vdev net_pcap0,...?\n");
	uint16_t port = 0;

	struct rte_mempool *pool = rte_pktmbuf_pool_create("mbufs", 8191, 256, 0,
			RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	if (!pool)
		rte_exit(EXIT_FAILURE, "mbuf pool: %s\n", rte_strerror(rte_errno));

	struct rte_eth_conf conf;
	memset(&conf, 0, sizeof(conf));
	if (rte_eth_dev_configure(port, 1, 1, &conf) < 0 ||
		rte_eth_rx_queue_setup(port, 0, 1024, rte_eth_dev_socket_id(port), NULL, pool) < 0 ||
		rte_eth_tx_queue_setup(port, 0, 1024, rte_eth_dev_socket_id(port), NULL) < 0 ||
		rte_eth_dev_start(port) < 0)
		rte_exit(EXIT_FAILURE, "port setup failed\n");

	uint64_t pkts = 0, bad = 0, dup = 0;
	uint64_t t0 = rte_get_tsc_cycles();
	struct rte_mbuf *bufs[BURST];
	for (unsigned idle = 0; idle < IDLE_POLLS;) {
		uint16_t n = rte_eth_rx_burst(port, 0, bufs, BURST);
		if (n == 0) {
			idle++;
			continue;
		}
		idle = 0;
		for (uint16_t i = 0; i < n; i++) {
			struct rte_mbuf *m = bufs[i];
			pkts++;
			uint8_t *p = rte_pktmbuf_mtod(m, uint8_t *);
			uint32_t len = rte_pktmbuf_data_len(m);
			struct rte_ether_hdr *eth = (struct rte_ether_hdr *)p;
			if (len < sizeof(*eth) + sizeof(struct rte_ipv4_hdr) ||
				eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4))
				goto drop;
			struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(eth + 1);
			uint32_t ihl = (ip->version_ihl & 0x0f) * 4;
			if (ip->next_proto_id != IPPROTO_UDP)
				goto drop;
			struct rte_udp_hdr *udp = (struct rte_udp_hdr *)((uint8_t *)ip + ihl);
			if (rte_be_to_cpu_16(udp->dst_port) != UDP_PORT)
				goto drop;
			uint8_t *payload = (uint8_t *)(udp + 1);
			uint32_t plen = len - (uint32_t)(payload - p);
			if (plen < sizeof(struct frag_hdr))
				goto drop;
			struct frag_hdr h;
			memcpy(&h, payload, sizeof(h)); /* x86 is little-endian, matching make_pcap.py */
			if (h.magic != MAGIC || h.frame_id >= MAX_FRAMES || h.seq >= MAX_SEGS ||
				h.length > plen - sizeof(h) || h.offset + h.length > h.frame_bytes)
				goto drop;

			struct frame *f = &frames[h.frame_id];
			if (!f->data) {
				f->data = calloc(1, h.frame_bytes);
				f->bytes = h.frame_bytes;
				f->nseg = h.nseg;
			}
			if (f->seen[h.seq]) {
				dup++;
			} else {
				memcpy(f->data + h.offset, payload + sizeof(h), h.length);
				f->seen[h.seq] = 1;
				f->got++;
				f->received += h.length;
			}
			rte_pktmbuf_free(m);
			continue;
		drop:
			bad++;
			rte_pktmbuf_free(m);
		}
	}
	double secs = (double)(rte_get_tsc_cycles() - t0) / rte_get_tsc_hz();

	/* Output: complete frames in frame_id order, each prefixed by its id (u32). */
	FILE *out = fopen(out_path, "wb");
	if (!out)
		rte_exit(EXIT_FAILURE, "cannot open %s\n", out_path);
	unsigned complete = 0, partial = 0;
	for (uint32_t id = 0; id < MAX_FRAMES; id++) {
		struct frame *f = &frames[id];
		if (!f->data)
			continue;
		if (f->got == f->nseg && f->received == f->bytes) {
			fwrite(&id, sizeof(id), 1, out);
			fwrite(f->data, 1, f->bytes, out);
			complete++;
		} else {
			printf("frame %u incomplete: %u/%u segments\n", id, f->got, f->nseg);
			partial++;
		}
		free(f->data);
	}
	fclose(out);

	printf("packets=%" PRIu64 " dropped=%" PRIu64 " duplicates=%" PRIu64 "\n", pkts, bad, dup);
	printf("frames complete=%u incomplete=%u -> %s\n", complete, partial, out_path);
	printf("elapsed %.3f s (includes %d idle polls at the end)\n", secs, IDLE_POLLS);

	rte_eth_dev_stop(port);
	rte_eth_dev_close(port);
	rte_eal_cleanup();
	return partial ? 1 : 0;
}
