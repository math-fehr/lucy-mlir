//===- Passes.h - Passes for the Obc dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_PASSES_H
#define OBC_PASSES_H

#include "Obc/ObcOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace obc {

std::unique_ptr<OperationPass<ObcMachine>> createDeclareRegLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createMachineLoweringPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Obc/ObcPasses.h.inc"

} // namespace obc
} // namespace mlir

#endif // OBC_OBCDIALECT_H
