add_mlir_dialect_library(MLIRObc
        ObcDialect.cpp
        ObcOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Obc

        DEPENDS
        MLIRObcOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
