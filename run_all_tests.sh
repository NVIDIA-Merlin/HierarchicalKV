#!/bin/bash

# Usage : `bash run_all_tests.sh`

# Keep test stdin separate from the NUL-delimited discovery stream.
has_tests=false
has_fail=false
while IFS= read -r -d '' file <&3
do
    has_tests=true
    echo "Executing $file ..."
    if ! "$file" 3<&-; then
        has_fail=true
    fi
done 3< <(find ./build/ -type f -name "*_test" -executable -print0)

# Bash exposes the process substitution's PID through $!.
if ! wait "$!"; then
    exit 1
fi

if [ "$has_tests" = false ]; then
    echo "No executable *_test binaries found in ./build/." >&2
    exit 1
fi

if [ "$has_fail" = true ]; then
    exit 1
fi
