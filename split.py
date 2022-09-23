"""
    split是fuse的反操作，把iter以factor为间隔分离成outer与inner
    两层迭代，增加循环层数，用于将循环操作分割为更小的子任务。事实上
    以CUDA为例，gridDim和blockDim都可以最多是三维，所以通过split
    可以产生新的维度用于绑定到grid和block上[3]。
"""
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, ), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)


print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))