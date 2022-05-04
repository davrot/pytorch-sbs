#!/bin/bash
for i in $(seq 1 1 999)
do
    echo $i
    /home/davrot/P3.10/bin/python3 test_it.py mnist.json $i
done