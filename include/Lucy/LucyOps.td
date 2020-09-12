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
    let verifier = [{ return ::verify(*this); }];
}

def LucyReturnOp : LucyOp<"return", [Terminator]> {
    let summary = "Return operation of a node.";
    let arguments = (ins LegalType:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def LucyFbyOp : LucyOp<"fby", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Get the previous value.";
    let arguments = (ins LegalType:$firstVal, LegalType:$nextVal);
    let results = (outs LegalType:$res);

    let assemblyFormat = [{
        $firstVal $nextVal attr-dict `:` type($firstVal)
    }];
}


#endif // LUCY_OPS
