//===- Passes.td - Obc dialect passes ----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_PASSES
#define OBC_PASSES

include "mlir/Pass/PassBase.td"
include "ObcBase.td"

def DeclareRegLoweringPass : Pass<"declare-reg-lowering", "ObcMachine"> {
    let summary = "Lower the declare_reg operations.";
    let constructor = "mlir::obc::createDeclareRegLoweringPass()";
}

#endif // OBC_PASSES