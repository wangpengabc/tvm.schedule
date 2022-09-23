"""
    unroll是一种常见的循环优化方法，减分支预测失败减少，如果循环体
    内语句没有数据相关，增加了并发执行的机会，也有利于指令流水线的
    调度[4]。
"""
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
B = te.placeholder((n, n), dtype=dtype, name='B')
C = te.compute((n, n), lambda i,j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

s[C].unroll(xi)
print(tvm.lower(s, [A, B, C], simple_mode=True))