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

Install Clang-format 9 (the version 9.0.1-12 is required) for Ubuntu:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - 
sudo add-apt-repository -u 'http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
sudo apt install clang-format-9
```

format all with:
```bash
find ./ -iname *.h -o -iname *.cpp -o -iname *.cc -o -iname *.cu -o -iname *.cuh | xargs clang-format-9 -i --style=file
```
