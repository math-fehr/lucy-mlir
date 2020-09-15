//===- LucyOps.cpp - Lucy dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lucy/LucyOps.h"
#include "Lucy/LucyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace lucy;

static void print(OpAsmPrinter &printer, LucyNode op) {
  // Print the node operation name
  printer << op.getOperationName() << ' ';

  // Print the node name
  printer.printSymbolName(
      op.getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue());

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

  // Print the node body
  printer.printRegion(op.body(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static ParseResult parseLucyNode(OpAsmParser &parser, OperationState &result) {
  // Parse the node"" name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return failure();
  };

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

  // Parse the node body.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, operands, types);
}

enum class TopoOpState {
  NotYetProcessed,
  BeingProcessed,
  Processed,
};

using TopoState = DenseMap<const Operation *, TopoOpState>;
using TopoResult = SmallVector<Operation *, 20>;

/// Process an operation after processing recursively its operands,
/// if the operation is not a follow by operation
bool topoSortNode(TopoState &state, TopoResult &res, Operation *op) {
  assert(state.count(op) == 1);

  // If the operation is being processed, then there is a cycle
  if (state[op] == TopoOpState::BeingProcessed)
    return false;
  // If the operation is processed, then it is already added to the result
  if (state[op] == TopoOpState::Processed)
    return true;

  // Start the processing of the operation
  state[op] = TopoOpState::BeingProcessed;

  // If the operation is a follow by operation,
  // then it does not create a dependency on the second operand
  auto fbyOp = dyn_cast<LucyFbyOp>(op);
  if (fbyOp) {
    auto *operand = op->getOperands().front().getDefiningOp();

    // Do the sort recursively on the first operand
    if (operand)
      if (!topoSortNode(state, res, operand))
        return false;
  } else {
    // Do the sort recursively on all operands
    for (auto operandValue : op->getOperands()) {
      auto *operand = operandValue.getDefiningOp();
      if (!operand)
        continue;

      if (!topoSortNode(state, res, operand)) {
        return false;
      }
    }
  }

  // When all operands are processed, we can add the operation to the result
  // list
  res.push_back(op);
  state[op] = TopoOpState::Processed;

  return true;
}

/// Topological sort of operations in a Lucy node
Optional<TopoResult> topoSortNode(LucyNode node) {
  auto &block = node.getRegion().getBlocks().front();
  auto ops = TopoState();
  auto result = TopoResult();

  for (auto &op : block)
    ops.insert({&op, TopoOpState::NotYetProcessed});

  for (auto &op : block)
    if (!topoSortNode(ops, result, &op))
      return {};

  return result;
}

LogicalResult verify(LucyNode node) {
  if (topoSortNode(node).hasValue())
    return success();
  return node.emitOpError("unexpected instantaneous loops");
}

namespace mlir {
namespace lucy {
#define GET_OP_CLASSES
#include "Lucy/LucyOps.cpp.inc"
} // namespace lucy
} // namespace mlir
