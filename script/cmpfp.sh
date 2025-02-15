#!/bin/sh

LS=$(ls $1 | sed -n "s/\([_A-Za-z0-9-]*\)-v[0-9]_csr.csv/\1/p" | uniq)
for f in $LS; do 
    echo " *** $f *** "; 
    python3 fpcmp.py "../bin/device-profile/yvector-gpu/$f-v1_csr.csv" "../bin/device-profile/yvector-gpu/$f-v3_csr.csv"; done