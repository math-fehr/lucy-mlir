add_mlir_dialect(LucyOps lucy)

set(LLVM_TARGET_DEFINITIONS LucyDialect.td)
mlir_tablegen(LucyInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LucyInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLucyInterfacesIncGen)
add_dependencies(mlir-headers MLIRLucyInterfacesIncGen)

add_mlir_doc(LucyDialect -gen-dialect-doc LucyDialect Lucy/)
add_mlir_doc(LucyOps -gen-op-doc LucyOps Lucy/)
