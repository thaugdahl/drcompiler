configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DRCOMPILER_BENCH_DEPENDS dr-opt)

# FileCheck and not are only available when building against an LLVM build tree
# (not an installed LLVM).  Guard so the Docker image can build without them.
if(TARGET FileCheck)
  list(APPEND DRCOMPILER_BENCH_DEPENDS FileCheck)
endif()
if(TARGET not)
  list(APPEND DRCOMPILER_BENCH_DEPENDS not)
endif()

# Not part of check-drcompiler: this is the behavioural corpus (A-G, regpress,
# runtime), run on demand.
add_lit_target(check-drcompiler-bench "Running drcompiler benchmark corpus"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DRCOMPILER_BENCH_DEPENDS}
)
