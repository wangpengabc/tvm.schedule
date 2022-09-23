"""
    use用于融合两个iter，将两层循环合并到一层，其返回值为iter类型，
    可以多次合并。
"""
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, ), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

s[B].fuse(ko, ki)
print(tvm.lower(s, [A, B], simple_mode=True))