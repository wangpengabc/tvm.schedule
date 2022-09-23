"""
    cache_read将tensor读入指定存储层次scope的cache，这个设计的意义在于显式利用
    现有计算设备的on-chip memory hierarchy。这个例子中，会先将A的数据load到
    shared memory中，然后计算B。在这里，我们需要引入一个stage的概念，一个op对
    应一个stage，也就是通过cache_read会新增一个stage。
"""

import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n, ), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("----------------")

AA = s.cache_read(A, "shared", [B])
print(tvm.lower(s, [A, B], simple_mode=True))

