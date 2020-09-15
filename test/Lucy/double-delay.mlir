// RUN: lucy-opt %s | lucy-opt

module {
    lucy.node @main(%i: i32) {
        %zero = constant 0 : i32
        %one = constant 1 : i32
        %fb1 = lucy.fby %one, %i : i32
        %fb2 = lucy.fby %zero, %fb1 : i32
        lucy.return %fb2 : i32
    }
}