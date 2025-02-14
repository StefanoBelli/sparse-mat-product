#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <dir>"
    exit 1
fi

dir="$1"

for h_file in "$dir"/*_hll.csv; do
    c_file="${h_file%_hll.csv}_csr.csv"
    
    if [ ! -f "$c_file" ]; then
        echo "missing $c_file"
        exit 1
    fi

    echo "comparing \"$c_file\" with \"$h_file\""

    line_number=0
    while IFS= read -r line_h || [ -n "$line_h" ]; do
        line_number=$((line_number + 1))
        printf "\tchecking line $line_number... "
        if ! IFS= read -r line_c <&3 || [ "$line_h" != "$line_c" ]; then
            echo "found diff in $c_file:$line_number"
            exit 1
        fi
        printf "ok\r"
    done < "$h_file" 3< "$c_file"
done

echo "all checks are passing!"