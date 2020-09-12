//===- LucyOps.cpp - Lucy dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lucy/LucyOps.h"
#include "Lucy/LucyDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace lucy;

static void print(OpAsmPrinter &p, LucyNode op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue());

  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static ParseResult parseLucyNode(OpAsmParser &parser, OperationState &result) {
  // Parse the node name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  };

  // Parse the node body.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, /*regionArgs*/ {}, /*argTypes*/ {});
}

namespace mlir {
namespace lucy {
#define GET_OP_CLASSES
#include "Lucy/LucyOps.cpp.inc"
} // namespace lucy
} // namespace mlir
