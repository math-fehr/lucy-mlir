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

def Lucy_Dialect : Dialect {
    let name = "lucy";
    let summary = "The high level dialect of a lucy program.";
    let cppNamespace = "lucy";
}

//===----------------------------------------------------------------------===//
// Base lucy operation definition.
//===----------------------------------------------------------------------===//

class Lucy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Lucy_Dialect, mnemonic, traits>;

#endif // LUCY_DIALECT
