#!/bin/bash

# Compares the corresponding files in two directories
# Usage: ./compare_two_directories <first_dir> <second_dir>

for file_full_path in $1/*; do
     filename=`basename $file_full_path`
     #echo 'Comparison Report for file: '$file_full_path
     cmp -l $2/$filename $1/$filename
     return_value=$?
     echo 'RETURN_VALUE: '$return_value
done

