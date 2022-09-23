"""
    reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用
    cache中的现有数据，减少反复载入载出的情况。注意，这里到底怎样的
    顺序是最优化的是一个很有趣的问题。以矩阵乘法为例，M, N, K三维，
    往往是将K放在最外层可以最大程度利用局部性。这个具体例子，具体探究。
"""
import tvm
from tvm import te

n = 1024
m = 1024

dtype = "float32"
A = te.placeholder((n, m), dtype=dtype, name='A')
l = te.reduce_axis((0, m), name='l')
B = te.compute((n,), lambda i: te.sum(A[i,l], axis=l), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

s[B].parallel(B.op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))