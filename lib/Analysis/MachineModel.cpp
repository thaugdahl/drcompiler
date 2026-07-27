//===- MachineModel.cpp - One machine, one description --------------------===//

#include "drcompiler/Analysis/MachineModel.h"
#include "drcompiler/Analysis/CpuCostModel.h"
#include "llvm/Support/raw_ostream.h"

using namespace drcompiler;

MachineModel MachineModel::fromJson(llvm::StringRef path) {
  MachineModel mm; // built-in defaults

  if (path.empty())
    return mm;

  // Reuse the existing JSON parser — there is exactly one cost-model file and
  // one parser for it.  Absent fields stay at their built-in defaults.
  CpuCostModel cm = CpuCostModel::loadFromFile(path);
  const CpuCacheJsonParams &c = cm.cacheParams();

  if (c.l1Size)
    mm.l1Size = static_cast<int64_t>(*c.l1Size);
  if (c.l2Size)
    mm.l2Size = static_cast<int64_t>(*c.l2Size);
  if (c.l3Size)
    mm.l3Size = static_cast<int64_t>(*c.l3Size);
  if (c.l1Latency)
    mm.l1Lat = *c.l1Latency;
  if (c.l2Latency)
    mm.l2Lat = *c.l2Latency;
  if (c.l3Latency)
    mm.l3Lat = *c.l3Latency;
  if (c.memLatency)
    mm.memLat = *c.memLatency;
  if (c.pageSize)
    mm.pageSize = static_cast<int64_t>(*c.pageSize);
  if (c.l2TlbEntries)
    mm.l2TlbEntries = static_cast<int64_t>(*c.l2TlbEntries);
  if (c.cacheLine)
    mm.cacheLine = static_cast<int64_t>(*c.cacheLine);
  if (c.llcSharers)
    mm.llcSharers = *c.llcSharers;

  // Vector-execution model (WP-G1).  Deriving `vl` from the model is gated on
  // the JSON having explicitly provided at least one of these, so the default
  // machine keeps the static vl option default (byte-identical pre-WP-G1 IR).
  const CpuArchJsonParams &a = cm.archParams();
  if (a.vectorBitsNative) {
    mm.vectorBitsNative = static_cast<int64_t>(*a.vectorBitsNative);
    mm.hasExplicitVectorModel = true;
  }
  if (a.vectorBitsArch) {
    mm.vectorBitsArch = static_cast<int64_t>(*a.vectorBitsArch);
    mm.hasExplicitVectorModel = true;
  }
  if (a.vecRegBudget) {
    mm.vecRegBudget = static_cast<int64_t>(*a.vecRegBudget);
    mm.hasExplicitVectorModel = true;
  }
  if (a.avx512FreqThrottle) {
    mm.avx512FreqThrottle = *a.avx512FreqThrottle;
    mm.hasExplicitVectorModel = true;
  }
  // Compute-roofline arm (WP-T2).  fmaUnits is a GEMM-model signal: it enables
  // the kernel-kind dispatch (gemmBlocking), so it sets hasExplicitGemmModel.
  if (a.fmaUnits) {
    mm.fmaUnits = *a.fmaUnits;
    mm.hasExplicitGemmModel = true;
  }
  // Leniency tunes the gemmBlocking tiling decision but does NOT by itself
  // enable it (that needs a GEMM-model signal above); map the value if present.
  if (a.llcTileLeniency)
    mm.llcTileLeniency = *a.llcTileLeniency;
  // Cross-field sanity: the native datapath cannot be wider than the encodable
  // vector (a partial JSON that sets only vector_bits_native above the default
  // vector_bits_arch, or an outright typo).  Correct + warn so preferredVector-
  // Elems never derives a vl wider than the register file.
  if (mm.hasExplicitVectorModel && mm.vectorBitsNative > mm.vectorBitsArch) {
    llvm::errs() << "drcompiler warning: arch.vector_bits_native ("
                 << mm.vectorBitsNative << ") > vector_bits_arch ("
                 << mm.vectorBitsArch << ") in '" << path
                 << "'; clamping native to arch\n";
    mm.vectorBitsNative = mm.vectorBitsArch;
  }

  // Thread / parallel-execution model (CROSSCUTTING.md III).  Gated on any field
  // being present so the default machine keeps the 1-thread, bandwidth-disabled
  // no-op (byte-identical).
  const CpuThreadJsonParams &t = cm.threadParams();
  auto markThread = [&] { mm.hasExplicitThreadModel = true; };
  if (t.activeThreads) {
    mm.thread.activeThreads = *t.activeThreads;
    markThread();
  }
  if (t.smtPerCore) {
    mm.thread.smtPerCore = *t.smtPerCore;
    markThread();
  }
  if (t.dramBytesPerCycle) {
    mm.thread.dramBytesPerCycle = *t.dramBytesPerCycle;
    markThread();
  }
  if (t.llcBytesPerCycle) {
    mm.thread.llcBytesPerCycle = *t.llcBytesPerCycle;
    markThread();
  }
  if (t.l1Shared) {
    mm.thread.l1Shared = *t.l1Shared;
    markThread();
  }
  if (t.l2Shared) {
    mm.thread.l2Shared = *t.l2Shared;
    markThread();
  }
  if (t.l3Shared) {
    mm.thread.l3Shared = *t.l3Shared;
    markThread();
  }
  if (t.exclusive) {
    mm.thread.exclusive = *t.exclusive;
    markThread();
  }

  // Register-file budgets (TRANSFORMER_KRNL_SPEC WP-T1).  CpuCostModel already
  // parses the `registers` block; bridge it into MachineModel so the GEMM
  // configurator and RegisterPressureAnalysis share one budget.  Presence of any
  // register budget is a GEMM-model signal (gates gemmBlocking via
  // hasExplicitGemmModel); WP-T2 adds arch.fmaUnits as a second trigger.
  const CpuRegisterJsonParams &r = cm.registerParams();
  auto markGemm = [&] { mm.hasExplicitGemmModel = true; };
  if (r.gpBudget) {
    mm.registers.gp = *r.gpBudget;
    markGemm();
  }
  if (r.fpBudget) {
    mm.registers.fp = *r.fpBudget;
    markGemm();
  }
  if (r.vecBudget) {
    mm.registers.vec = *r.vecBudget;
    markGemm();
  }
  if (r.predBudget) {
    mm.registers.pred = *r.predBudget;
    markGemm();
  }

  return mm;
}
