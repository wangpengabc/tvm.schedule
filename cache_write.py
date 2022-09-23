"""
    cache_write和cache_read对应，是先在shared memory
    中存放计算结果，最后将结果写回到global memory。当然
    在真实的场景中，我们往往是会将结果先放着register中，
    最后写回。
    Description: tvm/python/tvm/te/schedule.py
"""

import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n, ), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

AA = s.cache_write(B, "local")
print(tvm.lower(s, [A, B], simple_mode=True))

