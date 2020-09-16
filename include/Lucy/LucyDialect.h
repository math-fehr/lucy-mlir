//===- LucyDialect.h - Lucy dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LUCY_LUCYDIALECT_H
#define LUCY_LUCYDIALECT_H

#include "Lucy/Clocks.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace lucy {

#define GET_OP_CLASSES
#include "Lucy/LucyOpsDialect.h.inc"

#include "Lucy/LucyInterfaces.h.inc"

} // namespace lucy
} // namespace mlir

#endif // LUCY_LUCYDIALECT_H
