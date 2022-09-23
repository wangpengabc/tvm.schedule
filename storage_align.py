"""
    storage_align把stage对应的存储空间以factor为单位、以offset
    为偏置重新对齐，以避免GPU共享访问时的bank conflict，关于bank
    conflict可以参考[2]:
    https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
"""

import tvm
from tvm import te

n = 1024
factor = 100
offset = 8
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n, ), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)
AA = s.cache_read(A, "shared", [B])

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

s[AA].storage_align(AA.op.axis[0], factor, offset)
print(tvm.lower(s, [A, B], simple_mode=True))

