// RUN: obc-opt %s | obc-opt | FileCheck %s

module {
    obc.machine @main {
        %1 = obc.declare_reg : !obc.reg<i32>
        obc.body {
          // CHECK: obc.return %{{.*}} : i32
          %2 = obc.load %1 : (!obc.reg<i32>) -> i32
          obc.store %1 %2 : !obc.reg<i32> i32
          obc.return %2 : i32
        }
    }
}