# Transforms first: it defines the DRCompPassesIncGen tablegen target that the
# dialects and the libraries under lib/ depend on by name.
add_subdirectory(Transforms)
add_subdirectory(Dialect)
add_subdirectory(Analysis)
