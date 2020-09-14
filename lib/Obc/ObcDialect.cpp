//===- ObcDialect.cpp - Obc dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcDialect.h"
#include "Obc/ObcOps.h"
#include "Obc/ObcTypes.h"

using namespace mlir;
using namespace mlir::obc;

//===----------------------------------------------------------------------===//
// Obc dialect.
//===----------------------------------------------------------------------===//

void ObcDialect::initialize() {
  addTypes<RegisterType>();
  addOperations<
#define GET_OP_LIST
#include "Obc/ObcOps.cpp.inc"
      >();
}
