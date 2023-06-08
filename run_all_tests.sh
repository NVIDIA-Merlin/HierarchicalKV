#!/bin/bash

# Usage : `bash run_all_tests.sh`

# Search for all binary files that end with "test"
files=$(find ./build/ -type f -name "*_test" -executable)

# Execute each file found
has_fail=false
for file in $files
do
    echo "Executing $file ..."
    ./$file
    if ! [ $? -eq 0 ]; then
      has_fail=true
    fi
done

if [ "$has_fail" = true ] ; then
    exit 1
fi