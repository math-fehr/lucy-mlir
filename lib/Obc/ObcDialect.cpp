//===- ObcDialect.cpp - Obc dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcDialect.h"
#include "Obc/ObcOps.h"
#include "Obc/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::obc;

//===----------------------------------------------------------------------===//
// Obc dialect.
//===----------------------------------------------------------------------===//

void ObcDialect::initialize() {
  addTypes<StructType>();
  addOperations<
#define GET_OP_LIST
#include "Obc/ObcOps.cpp.inc"
      >();
}

/// from https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/

Type ObcDialect::parseType(DialectAsmParser &parser) const {
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<IntegerType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be an IntegerType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
  return Type();
}

/// Print an instance of a type registered to the toy dialect.
void ObcDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
