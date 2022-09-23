"""
    compute_at将当前的stage附着到目标stage的指定iter方向上，同时
    与目标stage采用相同的并行方式，在其内部完成当前stage的计算。往
    往compute_at会与cache_read和cache_write一起使用。
"""

import tvm
from tvm import te

n = 1024
factor = 100
offset = 8
dtype = "float32"
A = te.placeholder((n, ), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((1, ), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
BF = s.rfactor(B, ki)

tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
print(tvm.lower(s, [A, B], simple_mode=True))

