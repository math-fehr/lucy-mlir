//===- ObcOps.cpp - Obc dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcOps.h"
#include "Obc/ObcDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace obc;

namespace mlir {
namespace obc {
#define GET_OP_CLASSES
#include "Obc/ObcOps.cpp.inc"
} // namespace obc
} // namespace mlir
