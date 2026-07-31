//===- ArchHandlerRegistry.cpp - Name-based factory dispatch ---------------===//

#include "drcompiler/Analysis/ArchHandler.h"

#include "llvm/ADT/StringMap.h"

namespace drcompiler {

// Concrete factories from per-arch .cpp files.
std::unique_ptr<ArchHandler> makeGenericArchHandler();
std::unique_ptr<ArchHandler> makeX86_64_AVX2Handler();
std::unique_ptr<ArchHandler> makeX86_64_AVX512Handler();
std::unique_ptr<ArchHandler> makeARM_NeonHandler();
std::unique_ptr<ArchHandler> makeARM_SVEHandler();
std::unique_ptr<ArchHandler> makeAppleMSeriesHandler();

namespace {

using FactoryFn = std::unique_ptr<ArchHandler> (*)();

const llvm::StringMap<FactoryFn> &factories() {
  static const llvm::StringMap<FactoryFn> table = [] {
    llvm::StringMap<FactoryFn> m;
    m["generic"] = &makeGenericArchHandler;
    m["x86-64-avx2"] = &makeX86_64_AVX2Handler;
    m["x86-64-avx512"] = &makeX86_64_AVX512Handler;
    m["arm-neon"] = &makeARM_NeonHandler;
    m["arm-sve"] = &makeARM_SVEHandler;
    m["apple-m-series"] = &makeAppleMSeriesHandler;
    return m;
  }();
  return table;
}

} // namespace

std::unique_ptr<ArchHandler> ArchHandler::create(llvm::StringRef name) {
  if (name.empty())
    return makeGenericArchHandler();
  auto it = factories().find(name);
  if (it == factories().end())
    return makeGenericArchHandler();
  return it->second();
}

bool ArchHandler::isKnown(llvm::StringRef name) {
  return factories().find(name) != factories().end();
}

llvm::StringRef ArchHandler::pickHandlerForTriple(const llvm::Triple &t) {
  // Coarse heuristic: refined per-arch handlers only chosen via explicit
  // JSON.  Triple-only inference returns the architecture family default.
  switch (t.getArch()) {
  case llvm::Triple::x86_64:
    return "x86-64-avx2";
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    // aarch64-darwin is an Apple M-series (or A-series) part: wide front end,
    // 128 B cache line.  SVE is deliberately NOT inferred from a triple — it
    // is an optional feature of the architecture, not implied by `aarch64-*`,
    // so `arm-sve` must be named explicitly.
    if (t.isOSDarwin())
      return "apple-m-series";
    return "arm-neon";
  default:
    return "generic";
  }
}

} // namespace drcompiler
