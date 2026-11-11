#!/usr/bin/env bash
# Prepares a GitHub runner for the CPU model tests: frees disk, adds a swap file of $1 GB (the
# runner has 16 GB of memory) and makes the Hugging Face cache folder ($HF_HOME). Usage:
#   bash tools/create/runner-setup.sh 16
set -euo pipefail
swap_gb=${1:-16}
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL
sudo docker image prune -af > /dev/null 2>&1 || true
sudo swapoff -a || true
sudo rm -f /mnt/swapfile /swapfile
sudo fallocate -l "${swap_gb}G" /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile > /dev/null
sudo swapon /swapfile
sudo mkdir -p "${HF_HOME:-/mnt/hf}"
sudo chown "$(id -u):$(id -g)" "${HF_HOME:-/mnt/hf}"
[ -x /usr/bin/time ] || sudo apt-get install -y -qq time
free -g
df -h / /mnt 2>/dev/null || df -h /
