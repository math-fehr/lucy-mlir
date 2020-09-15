// RUN: lucy-opt %s -verify-diagnostics -split-input-file

module {
  lucy.node @main { // expected-error {{unexpected isntantaneous loops}}
    %0 = std.addi %1, %1 : i32
    %1 = std.addi %0, %0 : i32
  }
}