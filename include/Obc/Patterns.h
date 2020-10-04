//===- Patterns.h - Patterns for the Obc dialect ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_PATTERNS_H
#define OBC_PATTERNS_H

#include "Obc/ObcOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"


namespace mlir {
namespace obc {
#include "Obc/ObcPatterns.h.inc"
} // namespace obc
} // namespace mlir

#endif // OBC_PATTERNS_H
