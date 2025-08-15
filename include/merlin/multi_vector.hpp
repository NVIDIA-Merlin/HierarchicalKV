/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace nv {
namespace merlin {

/*
MultiVector supports:

1.Different types (any T1, T2, ...)

2.Each block of memory is 16-byte aligned

3.The first address of the i-th element can be retrieved using get<i>() (a
pointer of the correct type)

4.The total size of the entire multivector can be obtained

5.Large blocks of memory are allocated at once, with manual internal
partitioning (to improve memory locality)
*/
template <typename... Ts>
class MultiVector {
 public:
  static constexpr size_t Alignment = 16;

  template <typename... Lens, typename = typename std::enable_if<
                                  sizeof...(Lens) == sizeof...(Ts)>::type>
  explicit MultiVector(Lens... lens) {
    size_t tmp[] = {static_cast<size_t>(lens)...};
    for (size_t i = 0; i < sizeof...(Ts); ++i) {
      lengths_[i] = tmp[i];
    }
    compute_offsets();
  }

  ~MultiVector() {}

  template <size_t I>
  auto get(uint8_t* data) {
    using T = typename std::tuple_element<I, std::tuple<Ts...>>::type;
    return reinterpret_cast<T*>(data + offsets_[I]);
  }

  size_t length(size_t idx) const { return lengths_[idx]; }

  size_t offset(size_t idx) const { return offsets_[idx]; }

  size_t total_size() const { return total_size_; }

 private:
  std::array<size_t, sizeof...(Ts)> lengths_{};
  std::array<size_t, sizeof...(Ts)> offsets_{};
  size_t total_size_{0};

  constexpr size_t align_up(size_t n, size_t alignment) {
    return (n + alignment - 1) / alignment * alignment;
  }

  void compute_offsets() {
    size_t offset = 0;
    size_t idx = 0;

    (void)std::initializer_list<int>{
        (offset = align_up(offset, Alignment), offsets_[idx] = offset,
         offset += lengths_[idx] * sizeof(Ts), ++idx, 0)...};

    total_size_ = align_up(offset, Alignment);
  }
};

template <size_t I, typename... Ts>
auto get_vector(MultiVector<Ts...>& mv, uint8_t* data) {
  return mv.template get<I>(data);
}

}  // namespace merlin
}  // namespace nv