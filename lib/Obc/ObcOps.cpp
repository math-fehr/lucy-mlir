//===- ObcOps.cpp - Obc dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcOps.h"
#include "Obc/ObcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace obc;

//===----------------------------------------------------------------------===//
// Obc machine
//===----------------------------------------------------------------------===//

ObcStep ObcMachine::getStepOp() {
  auto *bodyReg = getBody();
  for (auto &op : *bodyReg) {
    if (auto stepOp = dyn_cast<ObcStep>(op)) {
      return stepOp;
    }
  }
  emitOpError(
      "Machine operations should have a step operation inside the body region");
}

bool ObcMachine::hasState() {
  auto *bodyReg = getBody();
  for (auto &op : *bodyReg) {
    if (isa<ObcStep>(op)) {
      return true;
    }
  }
  return false;
}

static void print(OpAsmPrinter &p, ObcMachine op) {
  // Print the machine operation name
  p << op.getOperationName() << ' ';

  // Print the machine name
  p.printSymbolName(
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue());

  assert(!op.getRegion().empty());

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

//===----------------------------------------------------------------------===//
// Obc machine step
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &printer, ObcStep op) {
  // Print the body operation name
  printer << op.getOperationName() << ' ';

  assert(!op.getRegion().empty());

  Block *body = op.getBody();
  auto args = body->getArguments();

  if (!args.empty()) {
    printer << "(";
    llvm::interleaveComma(args, printer, [&](auto arg) {
      printer << arg << " : " << arg.getType();
    });
    printer << ") ";
  }

  // Print the body
  printer.printRegion(op.body(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static ParseResult parseObcStep(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 10> operands;
  SmallVector<Type, 10> types;

  // Parse the operands
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::OperandType operand;
      Type type;

      if (parser.parseRegionArgument(operand) || parser.parseColonType(type))
        return failure();

      operands.push_back(operand);
      types.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));

    if (failed(parser.parseRParen()))
      return failure();
  }

  // Parse the region.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, operands, types);
}

//===----------------------------------------------------------------------===//
// Obc if/then/else
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ObcIte op) {
  // Print the operation name and the operand
  p << op.getOperationName() << ' ';
  p.printOperand(op.getOperand());

  // Print the then region
  p.printRegion(op.thenReg(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the else region
  p.printRegion(op.elseReg(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseObcIte(OpAsmParser &parser, OperationState &result) {
  // Parse the operand.
  OpAsmParser::OperandType operand;
  auto operandResult = parser.parseOperand(operand);
  if (failed(operandResult)) {
    return operandResult;
  }

  SmallVector<Value, 1> operands;
  auto int1 = IntegerType::get(1, result.getContext());
  auto loc = operand.location;
  if (failed(parser.resolveOperand(operand, int1, operands))) {
    return failure();
  }
  result.addOperands(operands);

  // Parse the then region.
  auto *thenReg = result.addRegion();
  auto thenRes =
      parser.parseRegion(*thenReg, /*regionArgs*/ {}, /*argTypes*/ {});
  if (failed(thenRes)) {
    return thenRes;
  }
  ObcIte::ensureTerminator(*thenReg, parser.getBuilder(), result.location);

  // Parse the else region.
  auto *elseReg = result.addRegion();
  auto elseRes =
      parser.parseRegion(*elseReg, /*regionArgs*/ {}, /*argTypes*/ {});
  if (failed(elseRes)) {
    return elseRes;
  }
  ObcIte::ensureTerminator(*elseReg, parser.getBuilder(), result.location);

  return success();
}

namespace mlir {
namespace obc {
#define GET_OP_CLASSES
#include "Obc/ObcOps.cpp.inc"
} // namespace obc
} // namespace mlir
