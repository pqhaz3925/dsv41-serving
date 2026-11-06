#!/bin/bash
# The ROCm images are tens of GB each and the root disk is small. Docker here stores image
# layers through containerd (/var/lib/containerd), so that directory is put on the NVMe array.
set -e
sudo systemctl stop docker docker.socket containerd
sudo mkdir -p /mnt/nvme/containerd
mountpoint -q /var/lib/containerd || { sudo rm -rf /var/lib/containerd/*; sudo mount --bind /mnt/nvme/containerd /var/lib/containerd; }
grep -q ' /var/lib/containerd ' /etc/fstab || echo '/mnt/nvme/containerd /var/lib/containerd none bind 0 0' | sudo tee -a /etc/fstab >/dev/null
sudo systemctl start containerd docker
sudo docker info 2>/dev/null | grep -i "driver-type"
df -h / /var/lib/containerd | tail -2
