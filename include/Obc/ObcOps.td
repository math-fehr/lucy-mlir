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

    let extraClassDeclaration = [{
      /// Has the machine declare_reg operations
      /// Those operations represent the state of the machine
      bool hasState();

      Block *getBody() { return &body().front(); }
      ObcStep getStepOp();
    }];

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcStep : ObcOp<"step", [Terminator, AutomaticAllocationScope, HasParent<"ObcMachine">]> {
    let summary = "The step function of a machine.";
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
      Block *getBody() { return &getRegion().front(); }
    }];

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

def ObcGetField : ObcOp<"get_field"> {
    let summary = "Get the field of a struct.";
    let results = (outs LegalType:$res);

    let arguments = (ins StructType:$input, I64Attr:$index);
    let results = (outs LegalType:$output);

    let assemblyFormat = [{
      $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def ObcSetField : ObcOp<"set_field"> {
    let summary = "Set the field of a struct.";
    let arguments = (ins StructType:$input, I64Attr:$index, LegalType:$value);

    let assemblyFormat = [{
      $input `[` $index `]` `,` $value attr-dict `:` type($input) `,` type($value)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def ObcIteTerminator : ObcOp<"ite_terminator", [Terminator]> {
    let assemblyFormat = [{
        attr-dict
    }];
}

def ObcIte : ObcOp<"ifthenelse", [SingleBlockImplicitTerminator<"ObcIteTerminator">]> {
    let summary = "An if/then/else operation.";
    let arguments = (ins I1:$input);
    let regions = (region SizedRegion<1>:$thenReg, SizedRegion<1>:$elseReg);

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ObcReturnOp : ObcOp<"return", [Terminator, HasParent<"ObcStep">]> {
    let summary = "Return operation of a machine.";
    let arguments = (ins LegalType:$input);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // OBC_OPS
