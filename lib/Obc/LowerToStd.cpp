//====- LowerToStd.cpp - Partial lowering from Toy to Std -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Obc operations to a combination of
// standard operations.
//
//===----------------------------------------------------------------------===//

#include "Obc/ObcDialect.h";
#include "Obc/ObcOps.h";
#include "Obc/Passes.h";
#include "Obc/Types.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdio>

using namespace mlir;
using namespace llvm;
using namespace obc;

namespace {

/// Conversion pattern with state utils memorized
template <typename OpTy>
struct StateConversionPattern : public ConversionPattern {
  StateConversionPattern(std::map<ObcDeclareReg, uint32_t> *declareRegsOrder,
                         ObcMachine *machineOp, BlockArgument *stateArg,
                         MLIRContext *ctx)
      : ConversionPattern(OpTy::getOperationName(), 1, ctx),
        declareRegsOrder(declareRegsOrder), machineOp(machineOp),
        stateArg(stateArg) {}

protected:
  /// Associate declare_reg instructions with index in the state struct
  std::map<ObcDeclareReg, uint32_t> *declareRegsOrder;
  ObcMachine *machineOp;
  BlockArgument *stateArg;
};

/// Remove loads of declare_reg and replace them with getFields
struct ReplaceLoadReg : public StateConversionPattern<LoadOp> {

  using StateConversionPattern::StateConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loadOp = cast<LoadOp>(op);
    auto loadOperand = loadOp.getOperand(0);
    auto declareRegOp = dyn_cast<ObcDeclareReg>(loadOperand.getDefiningOp());
    assert(declareRegOp);
    auto loc = declareRegOp.getLoc();

    auto stepOp = machineOp->getStepOp();
    auto stateArg = stepOp.getBody()->getArguments().back();
    auto index = declareRegsOrder->find(declareRegOp)->second;
    auto structType = stateArg.getType().cast<StructType>();
    auto fieldType = structType.getElementTypes()[index];

    auto newOp = rewriter.create<ObcGetField>(loc, fieldType, stateArg, index);
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

/// Remove stores of declare_reg and replace them with getFields
struct ReplaceStoreReg : public StateConversionPattern<StoreOp> {

  using StateConversionPattern::StateConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loadOp = cast<StoreOp>(op);
    auto loadOperand = loadOp.getOperand(1);
    auto value = loadOp.getOperand(0);
    auto declareRegOp = dyn_cast<ObcDeclareReg>(loadOperand.getDefiningOp());
    assert(declareRegOp);
    auto loc = declareRegOp.getLoc();

    auto stepOp = machineOp->getStepOp();
    auto stateArg = stepOp.getBody()->getArguments().back();
    auto index = declareRegsOrder->find(declareRegOp)->second;

    auto newOp = rewriter.create<ObcSetField>(declareRegOp.getLoc(), value,
                                              stateArg, index);
    rewriter.replaceOp(op, newOp.getOperation()->getResults());
    return success();
  }
};

/// Remove declare_reg operations
struct RemoveDeclareReg : public ConversionPattern {
  RemoveDeclareReg(MLIRContext *ctx)
      : ConversionPattern(ObcDeclareReg::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Remove ite operations
struct ReplaceIte : public ConversionPattern {
  ReplaceIte(MLIRContext *ctx)
      : ConversionPattern(ObcIte::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto *block = op->getBlock();
    auto *region = op->getParentRegion();
    auto iteOp = cast<ObcIte>(op);
    auto &thenReg = iteOp.thenReg();
    auto &elseReg = iteOp.elseReg();

    // Split the block at the if position
    auto *endBlock = block->splitBlock(op);

    // Clone the then region in between
    rewriter.cloneRegionBefore(thenReg, endBlock);
    auto *thenBlockBegin = block->getNextNode();
    auto *thenBlockEnd = endBlock->getPrevNode();

    // Clone the else region in between
    rewriter.cloneRegionBefore(elseReg, endBlock);
    auto *elseBlockBegin = thenBlockEnd->getNextNode();
    auto *elseBlockEnd = endBlock->getPrevNode();

    // Add the conditional branch
    rewriter.setInsertionPointToEnd(block);
    auto branchOp = rewriter.create<CondBranchOp>(
        loc, operands[0], thenBlockBegin, ArrayRef<Value>(), elseBlockBegin,
        ArrayRef<Value>());

    // Add the terminators on both branches
    rewriter.eraseOp(&thenBlockEnd->back());
    rewriter.eraseOp(&elseBlockEnd->back());
    rewriter.setInsertionPointToEnd(thenBlockEnd);
    rewriter.create<BranchOp>(loc, endBlock);
    rewriter.setInsertionPointToEnd(elseBlockEnd);
    rewriter.create<BranchOp>(loc, endBlock);

    // Remove the if operation
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/// Lower the ObcDeclareReg operations, by placing them in a struct, as operand
/// of the step function
struct DeclareRegLoweringPass
    : public DeclareRegLoweringPassBase<DeclareRegLoweringPass> {

  void runOnOperation() override {
    auto machineOp = getOperation();
    if (!machineOp.hasState()) {
      return;
    }
    auto stepOp = machineOp.getStepOp();
    auto *stepBody = stepOp.getBody();
    auto *machineBody = machineOp.getBody();

    // Get all declare_reg operations in the machine
    std::map<ObcDeclareReg, uint32_t> stateRegs;
    SmallVector<Type, 16> stateTypes;
    auto i = 0;
    for (auto &inst : *machineBody)
      if (auto declareRegOp = dyn_cast<ObcDeclareReg>(inst)) {
        stateRegs.insert({declareRegOp, i});
        i += 1;
        auto memRefType = declareRegOp.getType().cast<MemRefType>();
        stateTypes.push_back(memRefType.getElementType());
      }

    // If there is no ObcDeclareReg operations, we don't have anything to do
    if (stateRegs.empty())
      return;

    // Add the state to the step function inputs
    auto stateType = StructType::get(stateTypes);
    auto numArguments = stepBody->getNumArguments();
    auto state = stepBody->insertArgument(numArguments, stateType);

    OwningRewritePatternList patterns;
    ConversionTarget target(*machineOp.getContext());

    // We make illegal the declare_reg instructions,
    // and the load/store that had them as operands
    target.addLegalDialect<StandardOpsDialect, ObcDialect>();
    target.addIllegalOp<ObcDeclareReg>();
    target.addDynamicallyLegalOp<LoadOp>([](Operation *op) {
      return !isa<ObcDeclareReg>(
          cast<LoadOp>(op).getOperand(0).getDefiningOp());
    });
    target.addDynamicallyLegalOp<StoreOp>([](Operation *op) {
      return !isa<ObcDeclareReg>(
          cast<StoreOp>(op).getOperand(1).getDefiningOp());
    });

    // Remove load and stores that have ObcDeclareReg operands
    patterns.insert<ReplaceLoadReg, ReplaceStoreReg>(
        &stateRegs, &machineOp, &state, machineOp.getContext());
    // Remove declare_reg operations
    patterns.insert<RemoveDeclareReg>(machineOp.getContext());

    if (failed(applyFullConversion(machineOp, target, patterns)))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ObcMachine>> obc::createDeclareRegLoweringPass() {
  return std::make_unique<DeclareRegLoweringPass>();
}
