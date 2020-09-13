//===- ObcDialect.h - Obc dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_OBCDIALECT_H
#define OBC_OBCDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace obc {

#include "Obc/ObcOpsDialect.h.inc"

} // namespace obc
} // namespace mlir

#endif // OBC_OBCDIALECT_H