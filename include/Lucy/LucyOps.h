//===- LucyOps.h - Lucy dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LUCY_LUCYOPS_H
#define LUCY_LUCYOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace lucy {

#define GET_OP_CLASSES
#include "Lucy/LucyOps.h.inc"

} // namespace lucy
} // namespace mlir

#endif // LUCY_LUCYOPS_H
