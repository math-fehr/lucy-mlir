//===- LucyDialect.td - Lucy dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LUCY_DIALECT
#define LUCY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Lucy dialect definition.
//===----------------------------------------------------------------------===//

def LucyDialect : Dialect {
    let name = "lucy";
    let summary = "The high level dialect of a lucy program.";
    let cppNamespace = "lucy";
}

//===----------------------------------------------------------------------===//
// Lucy types
//===----------------------------------------------------------------------===//

def LegalType : AnyTypeOf<[I32, I1]>;

//===----------------------------------------------------------------------===//
// Base lucy operation definition.
//===----------------------------------------------------------------------===//

class LucyOp<string mnemonic, list<OpTrait> traits = []> :
        Op<LucyDialect, mnemonic, traits>;

#endif // LUCY_DIALECT
