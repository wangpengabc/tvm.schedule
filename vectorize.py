"""
    vectorize把iter方向上的循环迭代替换成ramp，从而通过SIMD指令
    实现数据的批量计算，并且只有在数据size为常数、且分割的iter为2
    的幂（即满足SIMD的计算数量）时才会发生替换，否则vectorize没
    有效果，是SIMD计算设备的常用schedule。
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

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")


s[C].vectorize(yi)
print(tvm.lower(s, [A, B, C], simple_mode=True))