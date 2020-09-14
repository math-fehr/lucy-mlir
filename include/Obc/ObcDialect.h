//===- ObcDialect.h - Obc dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_OBCDIALECT_H
#define OBC_OBCDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace obc {

class ObcDialect : public Dialect {
public:
  explicit ObcDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "obc"; }
  static StringRef getStencilProgramAttrName() { return "obc.program"; }

  static bool isStencilProgram(FuncOp funcOp) {
    return !!funcOp.getAttr(getStencilProgramAttrName());
  }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace obc
} // namespace mlir

#endif // OBC_OBCDIALECT_H
