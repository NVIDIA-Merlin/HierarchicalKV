#### C++
C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

HierarchicalKV uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

Install Clang-format (the version 18.1.3 is required) for Ubuntu:

```bash
sudo apt install clang-format-18
```

format all with:
```bash
find ./ \( -path ./tests/googletest -prune \) -o \( -iname *.h -o -iname *.cpp -o -iname *.cc -o -iname *.cu -o -iname *.cuh -o -iname *.hpp \) -print | xargs clang-format-18 -i --style=file

```
