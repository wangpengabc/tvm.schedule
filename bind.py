"""
    bind将iter绑定到block或thread的index上，从而把循环的任务
    分配到线程，实现并行化计算，这是针对CUDA后端最核心的部分。
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

s[B].bind(ko, te.thread_axis("blockIdx.x"))
s[B].bind(ki, te.thread_axis("threadIdx.x"))

print(tvm.lower(s, [A, B], simple_mode=True))