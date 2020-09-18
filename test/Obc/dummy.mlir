// RUN: obc-opt %s | obc-opt | FileCheck %s

module {
    obc.machine @main {
        %x_1 = obc.declare_reg : memref<i1>
        %x_2 = obc.declare_reg : memref<i32>
        obc.step {
          %o = std.alloca() : memref<i32>
          %x_3 = std.alloca() : memref<i32>
          %v = std.alloca() : memref<i32>
          %b = std.alloca() : memref<i1>
          %x_1_val = std.load %x_1[] : memref<i1>
          %x_2_val = std.load %x_2[] : memref<i32>
          // %2 = obc.load %1 : (!obc.reg<i32>) -> i32
          // obc.store %1 %2 : !obc.reg<i32> i32
          %2 = constant 2 : i32
          // CHECK: obc.return %{{.*}} : i32
          obc.return %2 : i32
        }
    }
}