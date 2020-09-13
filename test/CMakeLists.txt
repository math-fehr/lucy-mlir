configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LUCY_TEST_DEPENDS
        FileCheck count not
        lucy-opt
        lucy-translate
        obc-opt
        )

add_lit_testsuite(check-lucy "Running the lucy regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${LUCY_TEST_DEPENDS}
        )
set_target_properties(check-lucy PROPERTIES FOLDER "Tests")

add_lit_testsuites(LUCY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LUCY_TEST_DEPENDS})
