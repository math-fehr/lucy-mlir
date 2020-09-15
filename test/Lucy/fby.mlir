// RUN: lucy-opt %s | lucy-opt

module {
  lucy.node @main(%x: i1) {
    %zero = constant 0 : i32
    %one = constant 1 : i32
    %mone = constant -1 : i32
    %n1_1 = addi %n1, %one : i32
    %n1 = lucy.fby %zero, %n1_1 : i32
    %n2_1 = addi %n2, %one : i32
    %n2 = lucy.fby %one, %n2_1 : i32
    %b = cmpi "eq", %n1_1, %n2 : i32
    lucy.return %b : i1
  }
}