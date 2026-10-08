"""EE542 DOCA GPUNetIO test: two d7525 nodes (A30 GPU + ConnectX-6 Dx) on one LAN.

node0 (10.10.1.1) receives with DOCA GPUNetIO, node1 (10.10.1.2) sends.
On every boot each node runs cloudlab/boot.sh, which sets up the GPU, the NIC
and the DOCA container and writes its logs to /mydata/logs.

Instructions:
Wait until both nodes are ready, then read /mydata/logs/SUMMARY_<hostname>.txt
on each node. To test, run `bash /local/repository/cloudlab/gpunetio_test.sh receive`
on node0, then `bash /local/repository/cloudlab/gpunetio_test.sh send` on node1.
"""
# Repository-based profile: CloudLab reads profile.py from the top of the repo and
# clones the repo to /local/repository on every node.
#   https://docs.cloudlab.us/creating-profiles.html
#   https://docs.cloudlab.us/geni-lib.html
import geni.portal as portal
import geni.rspec.pg as pg

IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"
NODES = [("node0", "receiver", "10.10.1.1"), ("node1", "sender", "10.10.1.2")]

request = portal.context.makeRequestRSpec()
lan = request.LAN("lan")

for name, role, ip in NODES:
    node = request.RawPC(name)
    node.hardware_type = "d7525"
    node.disk_image = IMAGE

    iface = node.addInterface("if1")
    iface.addAddress(pg.IPv4Address(ip, "255.255.255.0"))
    lan.addInterface(iface)

    # Size 0GB = use all remaining disk space, same as small-lan's "Temp Filesystem Max Space".
    #   https://docs.cloudlab.us/advanced-storage.html
    bs = node.Blockstore(name + "-bs", "/mydata")
    bs.size = "0GB"
    bs.placement = "any"

    # Execute services run on every boot, so boot.sh also resumes after its own reboot.
    #   https://docs.cloudlab.us/geni-lib.html (section 8.7)
    node.addService(pg.Execute(shell="sh",
        command="sudo bash /local/repository/cloudlab/boot.sh %s > /dev/null 2>&1 &" % role))

portal.context.printRequestRSpec()
