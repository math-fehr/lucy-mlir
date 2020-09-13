get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRObc
        )
add_llvm_executable(obc-opt obc-opt.cpp)

llvm_update_compile_flags(obc-opt)
target_link_libraries(obc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(obc-opt)
