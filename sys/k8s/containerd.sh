sudo apt remove -y containerd
sudo apt update
sudo apt install containerd
sudo rm /etc/containerd/config.toml
sudo systemctl restart containerd