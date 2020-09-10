add_mlir_dialect_library(MLIRLucy
        LucyDialect.cpp
        LucyOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Lucy

        DEPENDS
        MLIRLucyOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
