get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRLucy
        )
add_llvm_executable(lucy-opt lucy-opt.cpp)

llvm_update_compile_flags(lucy-opt)
target_link_libraries(lucy-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(lucy-opt)
