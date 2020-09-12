// RUN: lucy-opt %s | lucy-opt | FileCheck %s

module {
    lucy.node @main {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = lucy.pre %{{.*}} : i32
        %1 = lucy.pre %0 : i32
        lucy.return %0 : i32
    }
}
