//===- ObcTypes.h - Obc types -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OBC_OBCTYPES_H
#define OBC_OBCTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

/// This class represents the internal storage of a single type
struct WrapperTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  /// A constructor for the type storage instance.
  WrapperTypeStorage(mlir::Type underlyingType)
      : underlyingType(underlyingType) {}

  // Check equality with the contained type
  bool operator==(const KeyTy &key) const { return key == underlyingType; }

  /// Define a construction function for the key type from a set of parameters.
  static KeyTy getKey(mlir::Type underlyingType) {
    return KeyTy(underlyingType);
  }

  /// Define a construction method for creating a new instance of this storage.
  static WrapperTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<WrapperTypeStorage>())
        WrapperTypeStorage(key);
  }

  mlir::Type getUnderlyingType() const {
    return underlyingType;
  }

private:
  /// The following field contains the contained type.
  mlir::Type underlyingType;
};

/// Type of a register. The register is parametric on the contained type
class RegisterType : public mlir::Type::TypeBase<RegisterType, mlir::Type,
                                                 WrapperTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of RegisterType given an element type.
  static RegisterType get(mlir::Type elementType) {
    mlir::MLIRContext *ctx = elementType.getContext();
    return Base::get(ctx, elementType);
  }

  /// Return the element type.
  mlir::Type getElementType() { return getImpl()->getUnderlyingType(); }
};

#endif // OBC_OBCTYPES_H
