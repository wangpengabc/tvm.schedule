"""
    reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用
    cache中的现有数据，减少反复载入载出的情况。注意，这里到底怎样的
    顺序是最优化的是一个很有趣的问题。以矩阵乘法为例，M, N, K三维，
    往往是将K放在最外层可以最大程度利用局部性。这个具体例子，具体探究。
"""
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
B = te.placeholder((n, n), dtype=dtype, name='B')
k = te.reduce_axis((0, n), name="k")
C = te.compute((n, n), lambda i, j: te.sum(A[i,k] * B[k,j], axis=k), name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))