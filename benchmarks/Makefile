#===----------------------------------------------------------------------===//
# Makefile — Memory fission benchmarks
#
# Compares two pipeline configurations:
#   baseline:  cgeist -O2 → dr-opt (raise-malloc-to-memref) → lower → clang -O2
#   fission:   cgeist -O2 → dr-opt (raise-malloc-to-memref, memory-fission) → lower → clang -O2
#
# Targets:
#   make all              Build all benchmark executables
#   make bench            Build + time all benchmarks with hyperfine
#   make bench-<name>     Build + time a single benchmark
#   make check            Verify baseline and fission produce identical output
#   make clean            Remove build artifacts
#
# Override tool paths:  make CLANG_RESOURCE_DIR=/usr/lib/clang/21
#===----------------------------------------------------------------------===//

# --- Tool paths (extracted from generated build/drcc) ---
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD_DIR   := $(PROJECT_DIR)/build

CGEIST        := $(shell sed -n 's/^CGEIST="$${CGEIST:-\(.*\)}"/\1/p' $(BUILD_DIR)/drcc)
DR_OPT        := $(shell sed -n 's/^DR_OPT="$${DR_OPT:-\(.*\)}"/\1/p' $(BUILD_DIR)/drcc)
MLIR_OPT      := $(shell sed -n 's/^MLIR_OPT="$${MLIR_OPT:-\(.*\)}"/\1/p' $(BUILD_DIR)/drcc)
MLIR_TRANSLATE := $(shell sed -n 's/^MLIR_TRANSLATE="$${MLIR_TRANSLATE:-\(.*\)}"/\1/p' $(BUILD_DIR)/drcc)
CLANG         := $(shell sed -n 's/^CLANG="$${CLANG:-\(.*\)}"/\1/p' $(BUILD_DIR)/drcc)

CLANG_RESOURCE_DIR ?= /usr/lib/clang/21

# --- Benchmarks ---
SRCS  := $(wildcard *.c)
NAMES := $(SRCS:.c=)
OUT   := out

BASELINES := $(addprefix $(OUT)/,$(addsuffix .baseline,$(NAMES)))
FISSIONS  := $(addprefix $(OUT)/,$(addsuffix .fission,$(NAMES)))

# --- Flags ---
CGEIST_FLAGS := -S --function='*' --raise-scf-to-affine --memref-fullrank \
                -resource-dir=$(CLANG_RESOURCE_DIR) -O2

LOWER_FLAGS := --lower-affine --convert-scf-to-cf --convert-cf-to-llvm \
               --convert-arith-to-llvm --convert-math-to-llvm \
               --convert-func-to-llvm --convert-index-to-llvm \
               --finalize-memref-to-llvm --reconcile-unrealized-casts

BASELINE_PIPELINE := builtin.module(raise-malloc-to-memref)
FISSION_PIPELINE  := builtin.module(raise-malloc-to-memref,memory-fission)

HYPERFINE_FLAGS ?= --warmup 2 --min-runs 5

# ===== Top-level targets =====================================================

.PHONY: all bench check clean

all: $(BASELINES) $(FISSIONS)

bench: $(BASELINES) $(FISSIONS)
	@for name in $(NAMES); do \
		echo ""; \
		echo "=== $$name ==="; \
		hyperfine $(HYPERFINE_FLAGS) \
			--command-name "baseline" "$(OUT)/$$name.baseline" \
			--command-name "fission"  "$(OUT)/$$name.fission"; \
	done

check: $(BASELINES) $(FISSIONS)
	@fail=0; \
	for name in $(NAMES); do \
		ob=$$($(OUT)/$$name.baseline 2>&1); \
		of=$$($(OUT)/$$name.fission 2>&1); \
		if [ "$$ob" = "$$of" ]; then \
			echo "OK  $$name"; \
		else \
			echo "MISMATCH $$name"; \
			echo "  baseline: $$ob"; \
			echo "  fission:  $$of"; \
			fail=1; \
		fi; \
	done; \
	exit $$fail

clean:
	rm -rf $(OUT)

# Per-benchmark bench target: make bench-sqrt_consumers
bench-%: $(OUT)/%.baseline $(OUT)/%.fission
	hyperfine $(HYPERFINE_FLAGS) \
		--command-name "baseline" "$(OUT)/$*.baseline" \
		--command-name "fission"  "$(OUT)/$*.fission"

# ===== Pipeline rules ========================================================

$(OUT):
	mkdir -p $@

# Step 1: C → MLIR (cgeist -O2 + DLTI fixup)
$(OUT)/%.mlir: %.c | $(OUT)
	$(CGEIST) $< $(CGEIST_FLAGS) -o $@ 2>/dev/null
	sed -i 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' $@

# Step 2a: baseline dr-opt passes
$(OUT)/%.baseline.opt.mlir: $(OUT)/%.mlir
	$(DR_OPT) --pass-pipeline='$(BASELINE_PIPELINE)' $< -o $@ 2>/dev/null

# Step 2b: fission dr-opt passes
$(OUT)/%.fission.opt.mlir: $(OUT)/%.mlir
	$(DR_OPT) --pass-pipeline='$(FISSION_PIPELINE)' $< -o $@ 2>/dev/null

# Step 3: lower to LLVM dialect
$(OUT)/%.llvm.mlir: $(OUT)/%.opt.mlir
	$(MLIR_OPT) $(LOWER_FLAGS) $< -o $@

# Step 4: MLIR → LLVM IR
$(OUT)/%.ll: $(OUT)/%.llvm.mlir
	$(MLIR_TRANSLATE) --mlir-to-llvmir $< -o $@

# Step 5: LLVM IR → executable
$(OUT)/%: $(OUT)/%.ll
	$(CLANG) -O2 $< -o $@ -lm

# Keep all intermediates for debugging
.SECONDARY:
