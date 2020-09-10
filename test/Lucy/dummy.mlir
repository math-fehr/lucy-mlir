// RUN: lucy-opt %s | lucy-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = lucy.pre %{{.*}} : i32
        %res = lucy.pre %0 : i32
        return
    }
}
