#!/bin/bash

# clear docker
sudo apt remove -y docker.* docker-*
sudo rm -rf /var/lib/docker /etc/docker
sudo rm /etc/apparmor.d/docker
sudo groupdel docker
sudo rm -rf /var/run/docker.sock

# clear container.d
sudo apt remove containerd

# clear K8s
sudo apt remove -y kubectl kubeadm kubectl
