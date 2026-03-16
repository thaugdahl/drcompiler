configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DRCOMPILER_TEST_DEPENDS
  dr-opt
  FileCheck
  not
)

add_lit_target(check-drcompiler "Running drcompiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DRCOMPILER_TEST_DEPENDS}
)
