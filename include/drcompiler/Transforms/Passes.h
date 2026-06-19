#ifndef DRCOMPILER_TRANSFORMS_PASSES_H
#define DRCOMPILER_TRANSFORMS_PASSES_H

#include "drcompiler/Transforms/AffineLoopFusion.h"
#include "drcompiler/Transforms/AffineLoopTile.h"
#include "drcompiler/Transforms/AffineRegisterBlock.h"
#include "drcompiler/Transforms/AffineStencilTimeTile.h"
#include "drcompiler/Transforms/ConvertParToOMP.h"
#include "drcompiler/Transforms/ConvertParToSCF.h"
#include "drcompiler/Transforms/DataRecomputation.h"
#include "drcompiler/Transforms/DrAffineLoopDistribute.h"
#include "drcompiler/Transforms/DrPinLiveOut.h"
#include "drcompiler/Transforms/LowerKrnlGlobal.h"
#include "drcompiler/Transforms/MathStrengthReduce.h"
#include "drcompiler/Transforms/MemoryFission.h"
#include "drcompiler/Transforms/ParBubbles.h"
#include "drcompiler/Transforms/PrintArchHandler.h"
#include "drcompiler/Transforms/PrintRegisterPressure.h"
#include "drcompiler/Transforms/RaiseMallocToMemRef.h"
#include "drcompiler/Transforms/ScalarReductionDemote.h"
#include "drcompiler/Transforms/ScalarReductionPromote.h"
#include "drcompiler/Transforms/TestReuseAnalysis.h"

namespace mlir {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PASSES_H
