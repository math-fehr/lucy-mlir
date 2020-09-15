// RUN: lucy-opt %s | lucy-opt

module {
  lucy.node @main(%ck: i1) {
    %zero = constant 0 : i32
    %one = constant 1 : i32
    %zero_ck = lucy.when %zero, %ck : i32
    %one_not_ck = lucy.when not %one, %ck : i32
    %res = lucy.merge %ck, %zero_ck, %one_not_ck : i32
    lucy.return %res : i32
  }
}