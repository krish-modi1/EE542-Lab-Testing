/*
 * DPDK receiver: read the packets in images.pcap through DPDK's pcap driver (no network card
 * needed), check the Ethernet/IPv4/UDP headers, and put the images back together.
 * Complete images are written to frames_rx.bin as [frame_id u32][150,528 image bytes].
 *
 * With DOCA GPUNetIO, a CUDA kernel would do this parsing and the network card would write
 * each payload straight into GPU memory. Here the CPU does it.
 *
 * Run: ./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
 *          --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null' -- frames_rx.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_udp.h>

#define MAGIC      0x542D0CA0u
#define UDP_PORT   5000
#define MAX_FRAMES 64
#define MAX_SEGS   256
#define BURST      32
#define IDLE_POLLS 200000 /* stop after this many empty polls in a row (end of file) */

/* Must match HEADER in make_pcap.py. */
struct __attribute__((packed)) frag_hdr {
	uint32_t magic, frame_id;
	uint16_t seq, nseg;
	uint32_t offset;
	uint16_t length, pad;
	uint32_t frame_bytes;
};

struct frame {
	uint8_t *data;
	uint32_t bytes, nseg, got;
	uint8_t seen[MAX_SEGS];
};

static struct frame frames[MAX_FRAMES];

/* Return the UDP payload of a packet for our port, or NULL if it is not one of ours. */
static uint8_t *udp_payload(struct rte_mbuf *m, uint32_t *plen)
{
	uint8_t *p = rte_pktmbuf_mtod(m, uint8_t *);
	uint32_t len = rte_pktmbuf_data_len(m);
	struct rte_ether_hdr *eth = (struct rte_ether_hdr *)p;
	if (len < sizeof(*eth) + sizeof(struct rte_ipv4_hdr) ||
	    eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4))
		return NULL;
	struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(eth + 1);
	if (ip->next_proto_id != IPPROTO_UDP)
		return NULL;
	struct rte_udp_hdr *udp = (struct rte_udp_hdr *)((uint8_t *)ip + (ip->version_ihl & 0x0f) * 4);
	if (rte_be_to_cpu_16(udp->dst_port) != UDP_PORT)
		return NULL;
	uint8_t *payload = (uint8_t *)(udp + 1);
	*plen = len - (uint32_t)(payload - p);
	return payload;
}

/* Copy one packet's chunk into its frame. Returns 0 if the packet was used. */
static int add_chunk(uint8_t *payload, uint32_t plen)
{
	struct frag_hdr h;
	if (plen < sizeof(h))
		return -1;
	memcpy(&h, payload, sizeof(h)); /* x86 is little-endian, same as make_pcap.py */
	if (h.magic != MAGIC || h.frame_id >= MAX_FRAMES || h.seq >= MAX_SEGS ||
	    h.length > plen - sizeof(h) || h.offset + h.length > h.frame_bytes)
		return -1;

	struct frame *f = &frames[h.frame_id];
	if (!f->data) {
		f->data = calloc(1, h.frame_bytes);
		f->bytes = h.frame_bytes;
		f->nseg = h.nseg;
	}
	if (!f->seen[h.seq]) { /* ignore duplicates */
		memcpy(f->data + h.offset, payload + sizeof(h), h.length);
		f->seen[h.seq] = 1;
		f->got++;
	}
	return 0;
}

int main(int argc, char **argv)
{
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "EAL init failed\n");
	const char *out_path = argc - ret > 1 ? argv[ret + 1] : "frames_rx.bin";
	if (rte_eth_dev_count_avail() == 0)
		rte_exit(EXIT_FAILURE, "no ports: pass --vdev 'net_pcap0,rx_pcap=images.pcap,...'\n");

	/* One port (the pcap file), one receive queue, one transmit queue. */
	uint16_t port = 0;
	struct rte_mempool *pool = rte_pktmbuf_pool_create("mbufs", 8191, 256, 0,
			RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	struct rte_eth_conf conf = {0};
	if (!pool || rte_eth_dev_configure(port, 1, 1, &conf) < 0 ||
	    rte_eth_rx_queue_setup(port, 0, 1024, rte_socket_id(), NULL, pool) < 0 ||
	    rte_eth_tx_queue_setup(port, 0, 1024, rte_socket_id(), NULL) < 0 ||
	    rte_eth_dev_start(port) < 0)
		rte_exit(EXIT_FAILURE, "port setup failed\n");

	/* Receive packets in bursts until the file runs out. */
	unsigned pkts = 0, dropped = 0;
	struct rte_mbuf *bufs[BURST];
	for (unsigned idle = 0; idle < IDLE_POLLS; idle++) {
		uint16_t n = rte_eth_rx_burst(port, 0, bufs, BURST);
		if (n > 0)
			idle = 0;
		for (uint16_t i = 0; i < n; i++) {
			uint32_t plen;
			uint8_t *payload = udp_payload(bufs[i], &plen);
			if (!payload || add_chunk(payload, plen) < 0)
				dropped++;
			pkts++;
			rte_pktmbuf_free(bufs[i]);
		}
	}

	/* Write every complete frame. */
	FILE *out = fopen(out_path, "wb");
	unsigned complete = 0, incomplete = 0;
	for (uint32_t id = 0; id < MAX_FRAMES; id++) {
		struct frame *f = &frames[id];
		if (!f->data)
			continue;
		if (f->got == f->nseg) {
			fwrite(&id, sizeof(id), 1, out);
			fwrite(f->data, 1, f->bytes, out);
			complete++;
		} else {
			printf("frame %u incomplete: %u/%u segments\n", id, f->got, f->nseg);
			incomplete++;
		}
		free(f->data);
	}
	fclose(out);

	printf("packets=%u dropped=%u\n", pkts, dropped);
	printf("frames complete=%u incomplete=%u -> %s\n", complete, incomplete, out_path);
	rte_eth_dev_stop(port);
	rte_eal_cleanup();
	return incomplete ? 1 : 0;
}
