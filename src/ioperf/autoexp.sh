#!/bin/bash

for i in {1..5}
do
  python3 controller.py
  mkdir -p data/run$i
  mv data/ioperf-* data/run$i/
done 
