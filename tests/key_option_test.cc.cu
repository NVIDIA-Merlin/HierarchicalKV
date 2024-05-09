#include <cstdint>
#include <gtest/gtest.h>

struct KeyOptions {
  uint64_t EMPTY_KEY;
  uint64_t RECLAIM_KEY;
  uint64_t LOCKED_KEY;

  __host__ __device__ KeyOptions(uint64_t emptyKey, uint64_t reclaimKey, uint64_t lockedKey)
      : EMPTY_KEY(emptyKey), RECLAIM_KEY(reclaimKey), LOCKED_KEY(lockedKey) {}

  virtual __host__ __device__ bool isReservedKey(uint64_t key) const = 0;
  virtual __host__ __device__ bool isVacantKey(uint64_t key) const = 0;
  virtual ~KeyOptions() {}
};

constexpr uint64_t RESERVED_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr uint64_t VACANT_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFE);

class DefaultKeyOptions : public KeyOptions {
 public:
  __host__ __device__ DefaultKeyOptions(uint64_t emptyKey = UINT64_C(0xFFFFFFFFFFFFFFFF),
                                        uint64_t reclaimKey = UINT64_C(0xFFFFFFFFFFFFFFFE),
                                        uint64_t lockedKey = UINT64_C(0xFFFFFFFFFFFFFFFD))
      : KeyOptions(emptyKey, reclaimKey, lockedKey) {}

  __host__ __device__ bool isReservedKey(uint64_t key) const override {
    return (RESERVED_KEY_MASK & (key)) == RESERVED_KEY_MASK;
  }

  __host__ __device__ bool isVacantKey(uint64_t key) const override {
    return (VACANT_KEY_MASK & (key)) == VACANT_KEY_MASK;
  }
};

class CustomKeyOptions : public DefaultKeyOptions {
 public:
  __host__ __device__ CustomKeyOptions(uint64_t emptyKey = UINT64_C(0xFFFFFFFFFFFFFFFA))  // Custom EMPTY_KEY
      : DefaultKeyOptions(emptyKey, UINT64_C(0xFFFFFFFFFFFFFFFE), UINT64_C(0xFFFFFFFFFFFFFFFD)) {}

  __host__ __device__ bool isReservedKey(uint64_t key) const override {
    return key == EMPTY_KEY || key == RECLAIM_KEY || key == LOCKED_KEY;
  }

  __host__ __device__ bool isVacantKey(uint64_t key) const override {
    return key == EMPTY_KEY || key == RECLAIM_KEY;
  }
};

void testKeyOptions() {
  DefaultKeyOptions opts;
  if (opts.isReservedKey(UINT64_C(0xFFFFFFFFFFFFFFFF))) {
    printf("Empty key is reserved\n");
  }
  if (!opts.isReservedKey(UINT64_C(0x1))) {
    printf("Non-reserved key is not reserved\n");
  }

  CustomKeyOptions customOpts;
  if (!customOpts.isReservedKey(UINT64_C(0xFFFFFFFFFFFFFFFF))) {
    printf("Empty key is no longer treated as reserved\n");
  }
  if (customOpts.isReservedKey(UINT64_C(0xFFFFFFFFFFFFFFFA))) {
    printf("New empty key is reserved\n");
  }
}

TEST(KeyOptionsTest, testKeyOptions) {
  testKeyOptions();
}