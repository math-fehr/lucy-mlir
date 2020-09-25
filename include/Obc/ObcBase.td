//===- ObcDialect.td - Obc dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_DIALECT
#define OBC_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Obc dialect definition.
//===----------------------------------------------------------------------===//

def ObcDialect : Dialect {
    let name = "obc";
    let summary = "The low level dialect of a lucy program.";
    let cppNamespace = "obc";

    let extraClassDeclaration = [{
        /// Parse an instance of a type registered to the toy dialect.
        mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

        /// Print an instance of a type registered to the toy dialect.
        void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
    }];

}

//===----------------------------------------------------------------------===//
// Obc types
//===----------------------------------------------------------------------===//

def LegalType : AnyTypeOf<[I32, I1]>;
def MemRefLegalType : MemRefOf<[I32, I1]>;

//===----------------------------------------------------------------------===//
// Base obc operation definition.
//===----------------------------------------------------------------------===//

class ObcOp<string mnemonic, list<OpTrait> traits = []> :
        Op<ObcDialect, mnemonic, traits> {
}

#endif // OBC_DIALECT
