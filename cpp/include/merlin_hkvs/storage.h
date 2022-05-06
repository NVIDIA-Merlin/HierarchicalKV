namespace hkvs{

template <class K, class V>
class Storage {
 public:
  /**
  * Move constructor with separate allocator. If the map being moved is being
  * modified concurrently, behavior is unspecified.
  *
  * @param other the map being moved
  * @param alloc the allocator instance to use with the map
  */
  Storage(Storage &&other, const Allocator &alloc)
      : hash_fn_(std::move(other.hash_fn_)),
        eq_fn_(std::move(other.eq_fn_)),
        buckets_(std::move(other.buckets_), alloc),
        old_buckets_(std::move(other.old_buckets_), alloc),
        all_locks_(alloc),
        num_remaining_lazy_rehash_locks_(
            other.num_remaining_lazy_rehash_locks()),
        minimum_load_factor_(other.minimum_load_factor()),
        maximum_hashpower_(other.maximum_hashpower()),
        max_num_worker_threads_(other.max_num_worker_threads())
  }

  virtual ~Storage () {}

  /**
   * Returns the number of elements in the storage.
   *
   * @return number of elements in the storage
   */
  size_t size() const {
    return static_cast<size_type>(0);
  }

  /** Returns the current capacity of the storage, that is, @ref bucket_count()
   * &times; @ref slot_per_bucket().
   *
   * @return capacity of storage
   */
  size_type capacity() const { return bucket_count() * slot_per_bucket(); }

  /**
   * Returns the percentage the storage is filled, that is, @ref size() &divide;
   * @ref capacity().
   *
   * @return load factor of the storage
   */
  double load_factor() const {
    return static_cast<double>(size()) / static_cast<double>(capacity());
  }

  /**
   * Returns the allocator associated with the map
   *
   * @return the associated allocator
   */
  allocator_type get_allocator() const { return buckets_.get_allocator(); }

  /**
   * Equivalent to calling @ref uprase_fn with a functor that modifies the
   * given value and always returns false (meaning the element is not removed).
   * The passed-in functor must implement the method <tt>void
   * operator()(mapped_type&)</tt>.
   */
  template <typename K, typename F, typename... Args>
  bool upsert(K &&key, F fn, Args &&... val) {
    return uprase_fn(
        std::forward<K>(key),
        [&fn](mapped_type &v) {
          fn(v);
          return false;
        },
        std::forward<Args>(val)...);
  }

  /**
   * Copies the value associated with @p key into @p val. Equivalent to
   * calling @ref find_fn with a functor that copies the value into @p val. @c
   * mapped_type must be @c CopyAssignable.
   */
  template <typename K>
  bool find(const K &key, mapped_type &val) const {
    return find_fn(key, [&val](const mapped_type &v) mustorage { val = v; });
  }

  /**
   * Erases the key from the storage. Equivalent to calling @ref erase_fn with a
   * functor that just returns true.
   */
  template <typename K>
  bool erase(const K &key) {
    return erase_fn(key, [](mapped_type &) { return true; });
  }

  /**
   * Removes all elements in the storage, calling their destructors.
   */
  void clear() {
    auto all_locks_manager = lock_all(normal_mode());
    cuckoo_clear();
  }

};

}