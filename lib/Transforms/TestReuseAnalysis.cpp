//===- TestReuseAnalysis.cpp - lit-test printer for ReuseAnalysis ---------===//
//
// Diagnostic pass: runs analyzeBandReuse on every top-level perfect band and
// emits FileCheck-able remarks describing the per-reference classification
// and the band-level reuse/eviction verdicts.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/TestReuseAnalysis.h"

#include "drcompiler/Analysis/ReuseAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_DRTESTREUSEANALYSISPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace drcompiler::reuse;

namespace {

StringRef kindName(ReuseKind k) {
  switch (k) {
  case ReuseKind::Invariant:
    return "inv";
  case ReuseKind::Spatial:
    return "spat";
  case ReuseKind::Streaming:
    return "stream";
  }
  llvm_unreachable("covered switch");
}

struct DrTestReuseAnalysisPass
    : public impl::DrTestReuseAnalysisPassBase<DrTestReuseAnalysisPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    for (auto forOp : fn.getOps<AffineForOp>()) {
      SmallVector<AffineForOp, 6> band;
      getPerfectlyNestedLoops(band, forOp);

      auto infoOr = analyzeBandReuse(band);
      if (failed(infoOr)) {
        forOp->emitRemark("reuse-analysis: band UNANALYZABLE");
        continue;
      }
      BandReuseInfo &info = *infoOr;

      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "reuse-analysis: band depth=" << info.band.size() << " trips=[";
      llvm::interleaveComma(info.tripCounts, os);
      os << "] footprint=" << info.footprintBytes(info.tripCounts)
         << " evictedReuse=[";
      for (unsigned l = 0, e = info.band.size(); l < e; ++l)
        os << (l ? "," : "")
           << (info.loopCarriesEvictedReuse(l, cacheBytes) ? 1 : 0);
      os << "] anyTemporal=" << (info.anyTemporalReuse() ? 1 : 0);
      forOp->emitRemark(buf);

      for (const RefGroup &g : info.groups) {
        std::string gbuf;
        llvm::raw_string_ostream gos(gbuf);
        gos << "reuse-analysis: group members=" << g.members.size()
            << " span=[";
        llvm::interleaveComma(g.span, gos);
        gos << "] carries=[";
        for (unsigned l = 0, nl = g.carriesReuse.size(); l < nl; ++l)
          gos << (l ? "," : "") << (g.carriesReuse[l] ? 1 : 0);
        gos << "]";
        info.refs[g.members.front()].op->emitRemark(gbuf);
      }

      for (unsigned r = 0, e = info.refs.size(); r < e; ++r) {
        const RefInfo &ref = info.refs[r];
        std::string rbuf;
        llvm::raw_string_ostream ros(rbuf);
        ros << "reuse-analysis: ref " << (ref.isWrite ? "store" : "load")
            << " x" << ref.multiplicity << " kinds=[";
        for (unsigned l = 0, nl = ref.kinds.size(); l < nl; ++l)
          ros << (l ? "," : "") << kindName(ref.kinds[l]);
        ros << "] iterFP=[";
        for (unsigned l = 0, nl = info.band.size(); l < nl; ++l)
          ros << (l ? "," : "") << info.refIterFootprint(r, l);
        ros << "]";
        ref.op->emitRemark(rbuf);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrTestReuseAnalysisPass() {
  return std::make_unique<DrTestReuseAnalysisPass>();
}
