#!/bin/bash

sudo apt update -y
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# install docker
sudo apt install -y docker.io containerd
sudo usermod -aG docker $USER

# install k8s
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add
sudo apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main"
sudo apt update
# sudo apt-get install -y kubeadm kubelet kubectl
#### We use the 1.17.6 because glusterfs is removed in the latest 1.26.0 version
sudo apt-get install -y --allow-downgrades kubeadm=1.26.0-00 kubelet=1.26.0-00 kubectl=1.26.0-00
sudo apt-get install -y --allow-downgrades kubeadm=1.17.6-00 kubelet=1.17.6-00 kubectl=1.17.6-00
sudo apt-mark hold kubeadm kubelet kubectl
sudo kubeadm version
sudo swapoff -a

# Optional: run docker through systemd
sudo su
sudo echo '{"exec-opts": ["native.cgroupdriver=systemd"]}' >> /etc/docker/daemon.json
sudo systemctl daemon-reload
sudo systemctl restart docker

# Optional: install helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
sudo chmod 700 get_helm.sh
./get_helm.sh


# if flannel failed try:
# edit /etc/kubernetes/manifests/kube-controller-manager.yaml
# at command ,add
# --allocate-node-cidrs=true
# --cluster-cidr=10.244.0.0/16
# then,reload kubelet