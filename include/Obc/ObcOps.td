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

def ObcDeclareReg : ObcOp<"declare_reg"> {
    let summary = "Declare a register.";
    let results = (outs Obc_Register:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def ObcLoad : ObcOp<"load"> {
    let summary = "Load the value of a register.";
    let arguments = (ins Obc_Register:$reg);
    let results = (outs LegalType:$res);

    // TODO: find a way to replace `functional-type(...)` by `type($reg)` in ODS
    let assemblyFormat = [{
        $reg attr-dict `:` functional-type($reg, results)
    }];
}

def ObcStore : ObcOp<"store"> {
    let summary = "Store a value on a register.";
    let arguments = (ins Obc_Register:$reg, LegalType:$value);

    // TODO: find a way to remove `type($reg)` in ODS
    let assemblyFormat = [{
        $reg $value attr-dict `:` type($reg) type($value)
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
