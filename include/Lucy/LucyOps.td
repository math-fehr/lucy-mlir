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
    let summary = "A computation node.";
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
      // Implement RegionKindInterface
      static RegionKind getRegionKind(unsigned index) {
        return RegionKind::Graph;
      }

      Block *getBody() { return &getRegion().front(); }
    }];

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let verifier = [{ return ::verify(*this); }];
}

def LucyReturnOp : LucyOp<"return", [Terminator, HasParent<"LucyNode">]> {
    let summary = "Return operation of a node.";
    let arguments = (ins LegalType:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def LucyWhenOp : LucyOp<"when", [NoSideEffect, AllTypesMatch<["res", "var"]>, HasParent<"LucyNode">]> {
    let summary = "Sample a stream on a clock";
    let arguments = (ins UnitAttr:$is_when_not, LegalType:$var, I1:$ck);
    let results = (outs LegalType:$res);

    let assemblyFormat = [{
        (`not` $is_when_not^)? $var `,` $ck attr-dict `:` type($var)
    }];
}

def LucyMergeOp : LucyOp<"merge", [NoSideEffect, AllTypesMatch<["res", "op1", "op2"]>, HasParent<"LucyNode">]> {
    let summary = "Merge two streams on a clock";
    let arguments = (ins I1:$ck, LegalType:$op1, LegalType:$op2);
    let results = (outs LegalType:$res);

    let assemblyFormat = [{
        $ck `,` $op1 `,` $op2 attr-dict `:` type($res)
    }];
}

def LucyFbyOp : LucyOp<"fby", [NoSideEffect, SameOperandsAndResultType, HasParent<"LucyNode">]> {
    let summary = "Get the previous value.";
    let arguments = (ins LegalType:$firstVal, LegalType:$nextVal);
    let results = (outs LegalType:$res);

    let assemblyFormat = [{
        $firstVal `,` $nextVal attr-dict `:` type($firstVal)
    }];
}

#endif // LUCY_OPS
