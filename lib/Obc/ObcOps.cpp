//===- ObcOps.cpp - Obc dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcOps.h"
#include "Obc/ObcDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace obc;

static void print(OpAsmPrinter &p, ObcMachine op) {
  // Print the machine operation name
  p << op.getOperationName() << ' ';

  // Print the machine name
  p.printSymbolName(
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue());

  // Print the machine body
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static ParseResult parseObcMachine(OpAsmParser &parser,
                                   OperationState &result) {
  // Parse the machine name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  };

  // Parse the machine body.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, /*regionArgs*/ {}, /*argTypes*/ {});
}

static void print(OpAsmPrinter &p, ObcBody op) {
  // Print the body operation name
  p << op.getOperationName() << ' ';

  // Print the body
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static ParseResult parseObcBody(OpAsmParser &parser, OperationState &result) {
  // Parse the region.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, /*regionArgs*/ {}, /*argTypes*/ {});
}

namespace mlir {
namespace obc {
#define GET_OP_CLASSES
#include "Obc/ObcOps.cpp.inc"
} // namespace obc
} // namespace mlir
