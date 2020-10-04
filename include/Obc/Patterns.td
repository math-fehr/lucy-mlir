//===- Patterns.td - Obc dialect patterns ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_PATTERNS
#define OBC_PATTERNS

include "ObcOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

def ObcToStdReturnPattern : Pat<(ObcReturnOp $arg), (ReturnOp $arg)>;

#endif // OBC_PATTERNS
