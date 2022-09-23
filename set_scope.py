"""
    set_scope指定stage计算结果所在的存储层次，为tensor选择最优的存储位置，
    适用于设置线程间的共享内存。事实上，set_scope是cache_read的子操作。
    个人理解： set_scope 会改变内存allocate的位置，即线程是否共享，可以对应
    生成的TIR程序理解。
"""
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n, ), lambda i: te.sum(A[i, k], axis=k), name='B')
C = te.compute((n, ), lambda i: B[i] + 10, name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, C], simple_mode=True))
print("----------------")

s[B].set_scope('shared')
print(tvm.lower(s, [A, B], simple_mode=True))

""" Generated Code
@main = primfn(A_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024], [])} {
  allocate(B: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (i: int32, 0, 1024) {
      B_1: Buffer(B, float32, [1024], [])[i] = 0f32
      for (k: int32, 0, 1024) {
        B_1[i] = (B_1[i] + A[((i*1024) + k)])
      }
    }
    for (i_1: int32, 0, 1024) {
      C[i_1] = (B_1[i_1] + 10f32)
    }
  }
}


----------------
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024], [])} {
  allocate(C: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (i: int32, 0, 1024) {
      B[i] = 0f32
      for (k: int32, 0, 1024) {
        B[i] = (B[i] + A[((i*1024) + k)])
      }
    }
    for (i_1: int32, 0, 1024) {
      C_1: Buffer(C, float32, [1024], [])[i_1] = (B[i_1] + 10f32)
    }
  }
}

"""