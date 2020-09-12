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
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def LucyNode : LucyOp<"node", [IsolatedFromAbove, RegionKindInterface]> {
    let summary = "A computation node";
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
      // Implement RegionKindInterface
      static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
    }];

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def LucyReturnOp : LucyOp<"return", [Terminator]> {
    let summary = "Return operation of a node.";
    let arguments = (ins I32:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def LucyFooOp : LucyOp<"pre", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Get the previous value.";
    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // LUCY_OPS
