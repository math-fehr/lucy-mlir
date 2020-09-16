//===- LucyOps.cpp - Lucy dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lucy/LucyOps.h"
#include "Lucy/Clocks.h"
#include "Lucy/LucyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
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
  auto &block = *node.getBody();
  auto ops = TopoState();
  auto result = TopoResult();

  for (auto &op : block)
    ops.insert({&op, TopoOpState::NotYetProcessed});

  for (auto &op : block)
    if (!topoSortNode(ops, result, &op))
      return {};

  return result;
}

FailureOr<ClockType> getResultClock(Value v, ClockTypeCtx &ctx) {
  auto it = ctx.find(v);

  if (it != ctx.end()) {
    // Copy the clock type
    auto result = it->second;
    return std::move(result);
  }

  if (auto *op = v.getDefiningOp()) {

    // If we are in an op that behaves differently on the clocks,
    // we called its specialized function
    if (auto modClock = dyn_cast<ModifyClock>(op))
      return modClock.getResultClock(ctx);

    // If we don't have any operands, then we are always on the base clock
    if (op->getNumOperands() == 0) {
      for (auto result : op->getResults()) {
        assert(ctx.count(result) == 0);
        ctx[result] = {};
      }
      return ClockType();
    }

    // If we have operands, we need to check that they all have the same clock
    // We may need to compute recursively their clocks
    auto operand = op->getOperand(0);
    auto clockOperandRes = getResultClock(operand, ctx);
    if (failed(clockOperandRes)) {
      return clockOperandRes;
    }
    auto clockOperand = clockOperandRes.getValue();

    // We check that all operands have the same clocks
    for (auto otherOperand : op->getOperands().drop_front(1)) {
      auto otherClockOperandRes = getResultClock(otherOperand, ctx);
      if (failed(otherClockOperandRes)) {
        return otherClockOperandRes;
      }
      auto otherClockOperand = otherClockOperandRes.getValue();

      // If the clocks are different, we output an error
      if (clockOperand != otherClockOperand) {
        return {op->emitOpError("operands have different clock types")};
      }
    }

    for (auto res : op->getResults()) {
      ctx.insert({res, clockOperand});
    }

    return std::move(clockOperand);
  }

  // Values that are not operations have the base clock
  return ClockType();
}

FailureOr<ClockType> LucyWhenOp::getResultClock(ClockTypeCtx &ctx) {
  // Get the clock of the first operand
  auto operand = getOperand(0);
  auto clockTypeRes = ::getResultClock(operand, ctx);
  if (failed(clockTypeRes))
    return clockTypeRes;
  auto clockType = clockTypeRes.getValue();

  // Get the clock of the second operand
  auto clockOperand = getOperand(1);
  auto clockClockTypeRes = ::getResultClock(clockOperand, ctx);
  if (failed(clockClockTypeRes))
    return clockClockTypeRes;
  auto clockClockType = clockClockTypeRes.getValue();

  if (clockType != clockClockType)
    return {emitOpError("operands have different clock types")};

  auto clock = getOperand(1);
  if (clockType.data.count(clock))
    return {emitOpError("Clock is already dependent on given clock")};

  clockType.data.insert({clock, !isWhenNot()});
  ctx.insert({*this, clockType});
  return std::move(clockType);
}

FailureOr<ClockType> LucyMergeOp::getResultClock(ClockTypeCtx &ctx) {
  // Get the clock of the clock operand
  auto ckOperand = getOperand(0);
  auto ckClockTypeRes = ::getResultClock(ckOperand, ctx);
  if (failed(ckClockTypeRes))
    return ckClockTypeRes;
  auto ckClockType = ckClockTypeRes.getValue();

  // Get the clock of the second operand
  auto operand1 = getOperand(1);
  auto clockTypeRes1 = ::getResultClock(operand1, ctx);
  if (failed(clockTypeRes1))
    return clockTypeRes1;
  auto clockType1 = clockTypeRes1.getValue();

  // Get the clock of the third operand
  auto operand2 = getOperand(2);
  auto clockTypeRes2 = ::getResultClock(operand2, ctx);
  if (failed(clockTypeRes2))
    return clockTypeRes2;
  auto clockType2 = clockTypeRes2.getValue();

  auto *it1 = clockType1.data.find(ckOperand);
  if (it1 == clockType1.data.end() || !it1->second)
    return {emitOpError("operand 2 should be sampled on operand 1")};

  auto *it2 = clockType2.data.find(ckOperand);
  if (it2 == clockType2.data.end() || it2->second)
    return {emitOpError(
        "operand 3 should be sampled on the negation of operand 1")};

  clockType1.data.erase(it1);
  clockType2.data.erase(it2);

  if ((clockType1 != clockType2) || (ckClockType != clockType1))
    return {
        emitOpError("operand 1, 2 and 3 should have the same clocks outside "
                    "the sampling on operand 1")};

  ctx.insert({getResult(), ckClockType});
  return std::move(ckClockType);
}

FailureOr<ClockType> LucyFbyOp::getResultClock(ClockTypeCtx &ctx) {
  // Get the clock of the immediate value
  auto operand = getOperand(0);
  auto clockTypeRes = ::getResultClock(operand, ctx);
  if (failed(clockTypeRes))
    return clockTypeRes;
  auto clockType = clockTypeRes.getValue();

  // Here we are already putting the resulting clock.
  // In case there is a loop when looking at the second operand,
  // we will not have to call this function again, since the expected clock
  // result will already be stored. We check only after that its value was
  // correct.
  ctx.insert({getResult(), clockType});

  // Get the clock of the immediate value
  auto operand2 = getOperand(1);
  auto clockTypeRes2 = ::getResultClock(operand2, ctx);
  if (failed(clockTypeRes2))
    return clockTypeRes2;
  auto clockType2 = clockTypeRes2.getValue();

  if (clockType != clockType2)
    return {emitOpError("both operands should have the same clocks")};

  return clockType;
}

// Check that the node is well-typed relative to clocks.
// This function will hang if there is an instantaneous cycle.
LogicalResult verifyClocks(LucyNode node) {
  ClockTypeCtx ctx = ClockTypeCtx({});
  for (auto &op : *node.getBody()) {
    for (auto res : op.getResults()) {
      auto clockRes = getResultClock(res, ctx);
      if (failed(clockRes)) {
        return clockRes;
      }
    }
  }
  return success();
}

LogicalResult verify(LucyNode node) {
  if (!topoSortNode(node).hasValue())
    return node.emitOpError("unexpected instantaneous loops");
  return verifyClocks(node);
}

namespace mlir {
namespace lucy {
#define GET_OP_CLASSES
#include "Lucy/LucyOps.cpp.inc"
} // namespace lucy
} // namespace mlir
