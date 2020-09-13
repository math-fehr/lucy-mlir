//===- ObcOps.td - Obc dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_OPS
#define OBC_OPS

include "ObcDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def ObcMachine : ObcOp<"machine", [IsolatedFromAbove]> {
    let summary = "A state machine.";
    let regions = (region SizedRegion<1>:$body);

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcReg : ObcOp<"reg"> {
    let summary = "A register.";
    let results = (outs LegalType:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def ObcBody : ObcOp<"body", [Terminator]> {
    let summary = "The body of a machine.";
    let regions = (region SizedRegion<1>:$body);

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcReturnOp : ObcOp<"return", [Terminator]> {
    let summary = "Return operation of a machine.";
    let arguments = (ins LegalType:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // OBC_OPS
