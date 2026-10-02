# Recipes use '>' instead of a tab so the file survives copy/paste and web editors.
.RECIPEPREFIX = >
PKGCONF ?= pkg-config
CFLAGS += -O2 -Wall $(shell $(PKGCONF) --cflags libdpdk)
LDLIBS += $(shell $(PKGCONF) --libs libdpdk)

dpdk_rx: dpdk_rx.c
> $(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
> rm -f dpdk_rx frames_rx.bin
