# Build the DPDK receiver. Recipes start with '>' instead of a tab (.RECIPEPREFIX).
.RECIPEPREFIX = >
CFLAGS += -O2 -Wall $(shell pkg-config --cflags libdpdk)
LDLIBS += $(shell pkg-config --libs libdpdk)

dpdk_rx: dpdk_rx.c
> $(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
> rm -f dpdk_rx frames_rx.bin
