//===- ObcOps.td - Obc dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_OPS
#define OBC_OPS

include "ObcBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def ObcMachine : ObcOp<"machine", [IsolatedFromAbove]> {
    let summary = "A state machine.";
    let regions = (region SizedRegion<1>:$body);

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcStep : ObcOp<"step", [Terminator, AutomaticAllocationScope, HasParent<"ObcMachine">]> {
    let summary = "The step function of a machine.";
    let regions = (region SizedRegion<1>:$body);

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcDeclareReg : ObcOp<"declare_reg", [HasParent<"ObcMachine">]> {
    let summary = "Declare a register.";
    let results = (outs MemRefLegalType:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def ObcReturnOp : ObcOp<"return", [Terminator, HasParent<"ObcStep">]> {
    let summary = "Return operation of a machine.";
    let arguments = (ins LegalType:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // OBC_OPS
