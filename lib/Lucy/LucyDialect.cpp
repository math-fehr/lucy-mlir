//===- LucyDialect.cpp - Lucy dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lucy/LucyDialect.h"
#include "Lucy/LucyOps.h"

using namespace mlir;
using namespace mlir::lucy;

//===----------------------------------------------------------------------===//
// Lucy dialect.
//===----------------------------------------------------------------------===//

#include "Lucy/LucyInterfaces.cpp.inc"

void LucyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Lucy/LucyOps.cpp.inc"
      >();
}
