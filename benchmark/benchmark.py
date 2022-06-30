import tensorflow.compat.v1 as tf
from merlin_kv import tensorflow as mkv
import time
import itertools, os
from tabulate import tabulate


def one_test(dim, items_num, device, test_times, maxval):
  sess_config = tf.ConfigProto(intra_op_parallelism_threads=4,
                               inter_op_parallelism_threads=4)
  sess_config.allow_soft_placement = False
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False
  if device == "/GPU:0":
    os.environ['CUDA_VISISBLE_DEVICES'] = "0"
  else:
    os.environ['CUDA_VISISBLE_DEVICES'] = ""
  with tf.Session(config=sess_config) as sess:
    with tf.device(device):
      random_keys = tf.random.uniform([items_num],
                                      minval=0,
                                      maxval=maxval,
                                      dtype=tf.int64,
                                      seed=None,
                                      name=None)
      de = mkv.get_variable("tf_benchmark",
                            tf.int64,
                            tf.float32,
                            devices=[device],
                            initializer=0.0,
                            dim=dim)
      default_vals_for_insert = tf.constant([[0.0] * dim] * items_num)
      lookup_op = de.lookup(random_keys)
      insert_op = de.upsert(random_keys,
                            default_vals_for_insert,
                            allow_duplicated_keys=False)
      size_op = de.size()
    sess.run(random_keys)
    start_time = time.time()
    for _ in range(test_times):
      sess.run(random_keys)
    # 用于修正 random_keys的生产时间
    random_time = (time.time() - start_time) / test_times
    sess.run(insert_op)
    start_time = time.time()
    for _ in range(test_times):
      sess.run(insert_op)
    insert_time = (time.time() - start_time) / test_times - random_time
    sess.run(lookup_op)
    start_time = time.time()
    for _ in range(test_times):
      sess.run(lookup_op)
    lookup_time = (time.time() - start_time) / test_times - random_time
    table_size = sess.run(size_op)
    sess.close()
  tf.reset_default_graph()
  return insert_time, lookup_time, table_size / 1000


# 避免rehash
os.environ['TF_HASHTABLE_INIT_SIZE'] = "33554432"
test_list = []
for dim, test_times, items_num in \
    itertools.product(
      [8, 64, 128],  [20, ], [1024, 8192, 16384, 32768, 65536, 131072, 1048576
                ]):
  maxval = items_num * test_times * 10
  upsert_cpu, lookup_cpu, size_cpu = one_test(dim, items_num, '/GPU:0',
                                              test_times, maxval)
  upsert_gpu, lookup_gpu, size_gpu = one_test(dim, items_num, '/GPU:0',
                                              test_times, maxval)
  test_list.append([
      dim,
      items_num,
      test_times,
      "{:.3f}".format(items_num / (upsert_cpu * 1e6)),
      "{:.3f}".format(items_num / (upsert_gpu * 1e6)),
      "{:.3f}".format(items_num / (lookup_cpu * 1e6)),
      "{:.3f}".format(items_num / (lookup_gpu * 1e6)),  #size_cpu, size_gpu
  ])
headers = [
    'dim',
    'keys num',
    'test_times',
    'CPU.upsert\n(M keys/s)',
    'GPU.upsert\n(M keys/s)',
    'CPU.lookup\n(M keys/s)',
    'GPU.lookup\n(M keys/s)',
    # 'CPU.size\n(Kilo items)', 'GPU.size\n(Kilo items)'
]
print(tabulate(test_list, headers, tablefmt="github"))
