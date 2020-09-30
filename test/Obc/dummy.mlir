// RUN: obc-opt %s | obc-opt

module {
    obc.machine @main {
        %x_1 = obc.declare_reg : memref<i1>
        %x_2 = obc.declare_reg : memref<i32>
        obc.step (%tick: i1, %top: i1) {
          %o = std.alloca() : memref<i32>
          %x_3 = std.alloca() : memref<i32>
          %v = std.alloca() : memref<i32>
          %b = std.alloca() : memref<i1>
          %x_1_val = std.load %x_1[] : memref<i1>
          %x_2_val = std.load %x_2[] : memref<i32>
          %false = constant 0 : i1
          %zero = constant 0 : i32
          %one = constant 1 : i32
          std.store %x_1_val, %b[] : memref<i1>
          std.store %false, %x_1[] : memref<i1>
          %b_val = std.load %b[] : memref<i1>
          obc.ifthenelse %top {
            std.store %one, %v[] : memref<i32>
          } {
            std.store %zero, %v[] : memref<i32>
          }
          %v_val = std.load %v[] : memref<i32>
          obc.ifthenelse %b_val {
            std.store %zero, %x_3[] : memref<i32>
          } {
            %x_3_temp = std.addi %x_2_val, %v_val : i32
            std.store %x_3_temp, %x_3[] : memref<i32>
          }
          %x_3_val = std.load %x_3[] : memref<i32>
          obc.ifthenelse %tick {
            std.store %v_val, %o[] : memref<i32>
          } {
            std.store %x_3_val, %o[] : memref<i32>
          }
          %o_val = std.load %o[] : memref<i32>
          std.store %o_val, %x_2[] : memref<i32>
          obc.return %o_val : i32
        }
    }
}