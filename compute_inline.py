"""
    compute_inline把独立的计算操作转化成内联函数形式，
    在使用到原计算结果时再调用内联函数完成运算，通过
    compute_inline来减少一个stage。
"""

import tvm
from tvm import te

n = 1024
k = 3
pad = 2
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
W = te.placeholder((k, k), dtype=dtype, name='W')
m = (n - k + 2*pad) + 1
Apad = te.compute((n+2*pad, n+2*pad),
                lambda yy, xx: te.if_then_else(
                    te.all(yy >= pad, yy < pad+n, xx >= pad, xx < pad+n),
                    A[yy - pad, xx - pad], te.const(0., "float32")),
                     name='Apad')

ry = te.reduce_axis((0, k), name="ry")
rx = te.reduce_axis((0, k), name="rx")

B = te.compute((m, m),
                lambda yy, xx:
                    te.sum(Apad[yy+ry, xx+rx] * W[ry, rx], 
                    axis=[ry, rx]),
                    name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))
print("----------------")

s[Apad].compute_inline()
print(tvm.lower(s, [A, W, B], simple_mode=True))

