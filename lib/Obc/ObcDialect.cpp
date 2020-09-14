//===- ObcDialect.cpp - Obc dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcDialect.h"
#include "Obc/ObcOps.h"
#include "Obc/ObcTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::obc;

//===----------------------------------------------------------------------===//
// Obc dialect.
//===----------------------------------------------------------------------===//

ObcDialect::ObcDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ObcDialect>()) {
  addTypes<RegisterType>();
  addOperations<
#define GET_OP_LIST
#include "Obc/ObcOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type parser.
//===----------------------------------------------------------------------===//

/// Print an instance of a type registered to the obc dialect.
void ObcDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  RegisterType regType = type.cast<RegisterType>();
  assert(regType);

  printer << "reg<" << regType.getElementType() << ">";
}

Type ObcDialect::parseType(mlir::DialectAsmParser &parser) const {

  // Parse `reg` `<`
  if (parser.parseKeyword("reg") || parser.parseLess())
    return Type();

  // Parse the underlying type
  mlir::Type elementType;
  if (parser.parseType(elementType))
    return Type();


  if (parser.parseGreater())
    return Type();
  return RegisterType::get(elementType);
}
