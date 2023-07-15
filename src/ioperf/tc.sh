#!/bin/bash

PRIMARY_INTERFACE=$(ip route | grep default | awk '{print $5}')

tc filter del dev $PRIMARY_INTERFACE parent 1: protocol ip prio 1
tc class del dev $PRIMARY_INTERFACE classid 1:10
tc qdisc del dev $PRIMARY_INTERFACE root

tc qdisc add dev $PRIMARY_INTERFACE root handle 1: htb default 10
tc class add dev $PRIMARY_INTERFACE parent 1:1 classid 1:10 htb rate ${1}mbit
tc filter add dev $PRIMARY_INTERFACE protocol ip parent 1:0 prio 1 handle 10 fw classid 1:10
iptables -t mangle -A OUTPUT -p tcp --sport 2049 -j MARK --set-mark 10

