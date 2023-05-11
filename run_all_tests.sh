#!/bin/bash

# Search for all binary files that end with "test"
files=$(find ./build/ -type f -name "*_test" -executable)

# Execute each file found
for file in $files
do
    echo "Executing $file ..."
    ./$file
done