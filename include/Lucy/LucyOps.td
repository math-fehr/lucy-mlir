//===- LucyOps.td - Lucy dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LUCY_OPS
#define LUCY_OPS

include "LucyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Lucy_FooOp : Lucy_Op<"pre", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Get the previous value.";
    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // LUCY_OPS
