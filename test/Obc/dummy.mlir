// RUN: obc-opt %s | obc-opt | FileCheck %s

module {
    obc.machine @main {
        %1 = obc.reg : i32
        %2 = obc.reg : i1
        obc.body {
          // CHECK: obc.return %{{.*}} : i32
          obc.return %1 : i32
        }
    }
}