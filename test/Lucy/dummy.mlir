// RUN: lucy-opt %s | lucy-opt | FileCheck %s

module {
    lucy.node @main {
        %0 = constant 0 : i32
        %1 = constant 1 : i32
        // CHECK: %{{.*}} = lucy.fby %{{.*}} %{{.*}} : i32
        %2 = lucy.fby %0, %1 : i32
        lucy.return %2 : i32
    }
}