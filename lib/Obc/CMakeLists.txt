add_mlir_dialect_library(MLIRObc
        ObcDialect.cpp
        ObcOps.cpp
        LowerToStd.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Obc

        DEPENDS
        MLIRObcOpsIncGen
        MLIRObcPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
