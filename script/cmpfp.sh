#!/bin/sh

LS=$(ls $1/yvector-serial | sed -n "s/\([_A-Za-z0-9-]*\)_csr_v[0-9]_fp64.csv/\1/p" | uniq)
i=1
tot=$(echo $LS | wc -w)
for f in $LS; do 
    echo " *** $f $i / $tot *** " 
    python3 fpcmp.py "$1/yvector-serial/${f}_csr_v1_fp64.csv" "$1/yvector-gpu/${f}_csr_v3_fp64.csv"
    i=$(( $i + 1 ))
done