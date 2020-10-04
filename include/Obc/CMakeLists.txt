add_mlir_dialect(ObcOps obc)
add_mlir_doc(ObcBase -gen-dialect-doc ObcBase Obc/)
add_mlir_doc(ObcOps -gen-op-doc ObcOps Obc/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(ObcPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRObcPassesIncGen)

set(LLVM_TARGET_DEFINITIONS Patterns.td)
mlir_tablegen(ObcPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRObcPatternsIncGen)
