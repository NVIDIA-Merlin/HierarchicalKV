#!/bin/bash

# CPU-only regression fixtures for the test launcher's shell control flow.
# An optional launcher path allows the same checks to run against an old version.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher=${1:-"$script_dir/../run_all_tests.sh"}
launcher=$(cd -- "$(dirname -- "$launcher")" && pwd)/$(basename -- "$launcher")
fixture_root=$(mktemp -d)
trap 'rm -rf -- "$fixture_root"' EXIT
failures=0
find_path=$PATH
launcher_stdin=/dev/null

check()
{
    local description=$1
    shift
    if "$@"; then
        printf 'PASS: %s\n' "$description"
    else
        printf 'FAIL: %s\n' "$description" >&2
        cat -- "$case_dir/output.log" >&2
        failures=$((failures + 1))
    fi
}

run_launcher()
{
    if (cd -- "$case_dir" && PATH="$find_path" bash "$launcher" <"$launcher_stdin") >"$case_dir/output.log" 2>&1; then
        status=0
    else
        status=$?
    fi
}

case_dir="$fixture_root/missing"
mkdir -p -- "$case_dir"
run_launcher
check 'missing build fails' test "$status" -ne 0

case_dir="$fixture_root/empty"
mkdir -p -- "$case_dir/build"
run_launcher
check 'empty build fails' test "$status" -ne 0

case_dir="$fixture_root/nonexecutable"
mkdir -p -- "$case_dir/build"
printf '#!/bin/bash\nexit 0\n' >"$case_dir/build/nonexecutable_test"
run_launcher
check 'build with no executable tests fails' test "$status" -ne 0

case_dir="$fixture_root/success"
mkdir -p -- "$case_dir/build"
printf '#!/bin/bash\nprintf "success\\n" >> trace\n' >"$case_dir/build/success_test"
chmod +x -- "$case_dir/build/success_test"
run_launcher
check 'successful binary passes' test "$status" -eq 0
check 'successful binary executes' test "$(cat "$case_dir/trace" 2>/dev/null || true)" = success

case_dir="$fixture_root/aggregate"
mkdir -p -- "$case_dir/build"
cat >"$case_dir/build/first_test" <<'EOF'
#!/bin/bash
if [ ! -e trace ]; then
    printf 'fail\n' >> trace
    exit 17
fi
printf 'pass\n' >> trace
EOF
cp -- "$case_dir/build/first_test" "$case_dir/build/second_test"
chmod +x -- "$case_dir/build/first_test" "$case_dir/build/second_test"
run_launcher
check 'failed binary fails launcher' test "$status" -ne 0
check 'later binary executes after failure' test "$(cat "$case_dir/trace" 2>/dev/null || true)" = $'fail\npass'

case_dir="$fixture_root/discovery_fd"
mkdir -p -- "$case_dir/build"
cat >"$case_dir/build/first_test" <<'EOF'
#!/bin/bash
if [ ! -e trace ]; then
    printf 'first-passes\n' >> trace
    { cat <&3 >/dev/null; } 2>/dev/null || true
    exit 0
fi
printf 'second-fails\n' >> trace
exit 9
EOF
cp -- "$case_dir/build/first_test" "$case_dir/build/second_test"
chmod +x -- "$case_dir/build/first_test" "$case_dir/build/second_test"
run_launcher
check 'child cannot consume discovery and hide a failing binary' test "$status" -ne 0
check 'every discovered binary executes despite child descriptor reads' test "$(cat "$case_dir/trace" 2>/dev/null || true)" = $'first-passes\nsecond-fails'

case_dir="$fixture_root/stdin"
mkdir -p -- "$case_dir/build"
cat >"$case_dir/build/first_test" <<'EOF'
#!/bin/bash
payload=$(cat)
printf 'ran:%s\n' "$payload" >> trace
EOF
cp -- "$case_dir/build/first_test" "$case_dir/build/second_test"
chmod +x -- "$case_dir/build/first_test" "$case_dir/build/second_test"
printf 'caller input\n' >"$case_dir/input"
launcher_stdin="$case_dir/input"
run_launcher
check 'test stdin preserves caller input and cannot consume discovery' test "$status:$(cat "$case_dir/trace" 2>/dev/null || true)" = $'0:ran:caller input\nran:'
launcher_stdin=/dev/null

case_dir="$fixture_root/selection"
mkdir -p -- "$case_dir/build/directory_test"
printf '#!/bin/bash\nprintf "selected\\n" >> trace\n' >"$case_dir/build/selected_test"
printf '#!/bin/bash\nprintf "unexpected\\n" >> trace\n' >"$case_dir/build/ignored"
cp -- "$case_dir/build/ignored" "$case_dir/build/nonexecutable_test"
chmod +x -- "$case_dir/build/selected_test" "$case_dir/build/ignored"
run_launcher
check 'only executable *_test files selected' test "$status:$(cat "$case_dir/trace" 2>/dev/null || true)" = '0:selected'

case_dir="$fixture_root/whitespace"
binary_dir="$case_dir/build/sub dir"$'\t\n'"more"
binary_path="$binary_dir/white space"$'\t\n'"_test"
mkdir -p -- "$binary_dir"
printf '#!/bin/bash\nprintf "whitespace\\n" >> trace\n' >"$binary_path"
chmod +x -- "$binary_path"
run_launcher
check 'spaces, tabs, and newlines in paths pass' test "$status:$(cat "$case_dir/trace" 2>/dev/null || true)" = '0:whitespace'

case_dir="$fixture_root/discovery"
mkdir -p -- "$case_dir/build" "$case_dir/tools"
printf '#!/bin/bash\nprintf "discovered\\n" >> trace\n' >"$case_dir/build/discovered_test"
chmod +x -- "$case_dir/build/discovered_test"
printf '#!/bin/bash\nexit 23\n' >"$case_dir/tools/find"
chmod +x -- "$case_dir/tools/find"
find_path="$case_dir/tools:$PATH"
run_launcher
check 'discovery error without output fails' test "$status" -ne 0
printf '#!/bin/bash\nprintf "./build/discovered_test\\0"\nexit 23\n' >"$case_dir/tools/find"
run_launcher
check 'discovery error after output fails' test "$status" -ne 0
check 'partially discovered binary executes' test "$(cat "$case_dir/trace" 2>/dev/null || true)" = discovered

if [ "$failures" -ne 0 ]; then
    printf '%s launcher regression checks failed.\n' "$failures" >&2
    exit 1
fi
printf 'All launcher regression checks passed.\n'
