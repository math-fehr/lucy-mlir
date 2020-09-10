//===- LucyOps.cpp - Lucy dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lucy/LucyOps.h"
#include "Lucy/LucyDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace lucy {
#define GET_OP_CLASSES
#include "Lucy/LucyOps.cpp.inc"
} // namespace lucy
} // namespace mlir
