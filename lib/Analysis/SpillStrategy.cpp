//===- SpillStrategy.cpp - Name <-> enum conversions ----------------------===//

#include "drcompiler/Analysis/SpillStrategy.h"

namespace drcompiler {

SpillStrategy parseSpillStrategy(llvm::StringRef name) {
  if (name == "sum-excess")
    return SpillStrategy::SumExcess;
  if (name == "graph-color")
    return SpillStrategy::GraphColor;
  // Empty / unknown / "excess-hot" -> default.
  return SpillStrategy::ExcessHot;
}

llvm::StringRef spillStrategyName(SpillStrategy s) {
  switch (s) {
  case SpillStrategy::ExcessHot:
    return "excess-hot";
  case SpillStrategy::SumExcess:
    return "sum-excess";
  case SpillStrategy::GraphColor:
    return "graph-color";
  }
  return "excess-hot";
}

} // namespace drcompiler
