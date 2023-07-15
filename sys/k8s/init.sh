#!/bin/bash

# OS: Ubuntu 20.04.5 LTS

sudo kubeadm init --pod-network-cidr=10.244.0.0/16 
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# print token
sudo kubeadm token create --print-join-command

# install KubeRouter CNI
kubectl apply -f https://raw.githubusercontent.com/cloudnativelabs/kube-router/master/daemonset/kubeadm-kuberouter.yaml

# install Cilium CNI
helm repo add cilium https://helm.cilium.io/
# helm install cilium cilium/cilium --version 1.13.3 \
#   --namespace kube-system \
#   --set bandwidthManager.enabled=true

helm install cilium cilium/cilium --version 1.13.3 \
  --namespace kube-system \
  --set bandwidthManager.enabled=true
