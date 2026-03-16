# GenerateDrcc.cmake – replaces the @DROPT_EXECUTABLE@ placeholder
# with the real dr-opt path after the build-system generator expression
# has been resolved.
file(READ "${INPUT_FILE}" content)
string(REPLACE "%%DROPT_EXECUTABLE%%" "${DROPT}" content "${content}")
file(WRITE "${OUTPUT_FILE}" "${content}")
# Make executable.
file(CHMOD "${OUTPUT_FILE}" PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE)
