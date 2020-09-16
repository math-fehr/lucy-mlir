//===- Clocks.h - Lucy clocks -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LUCY_CLOCKS_H
#define LUCY_CLOCKS_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

struct ClockType {
  llvm::SmallMapVector<mlir::Value, bool, 4> data;

  bool operator==(const ClockType &ct) const {
    if (data.size() != ct.data.size())
      return false;

    for (auto it1 : ct.data) {
      const auto *it2 = ct.data.find(it1.first);
      if (it2 == ct.data.end())
        return false;
      if (it1.second != it2->second)
        return false;
    }

    return true;
  }

  bool operator!=(const ClockType &ct) const { return !(*this == ct); }

  void print(llvm::raw_ostream &OS) const {
    OS << "{ ";
    for (auto it : data) {
      OS << it.first << " " << it.second << ", ";
    }
    OS << "}";
  }

  LLVM_DUMP_METHOD void dump() const {
    print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
};

using ClockTypeCtx = llvm::DenseMap<mlir::Value, ClockType>;
/*
// Check equality between the two clock types
inline bool clockTypeEq(const ClockType &c1, const ClockType &c2) {
  if (c1.size() != c2.size())
    return false;

  for (auto it1 : c1) {
    const auto *it2 = c2.find(it1.first);
    if (it2 == c1.end())
      return false;
    if (it1.second != it2->second)
      return false;
  }

  return true;
}*/

#endif // LUCY_CLOCKS_H
