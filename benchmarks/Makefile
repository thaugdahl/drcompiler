#===----------------------------------------------------------------------===//
# Makefile — Memory fission benchmarks
#
# Compares pipeline configurations:
#   baseline:    cgeist -O2 → dr-opt (raise-malloc-to-memref) → lower → clang -O2
#   fission:     cgeist -O2 → dr-opt (raise-malloc-to-memref, memory-fission) → lower → clang -O2
#   fission-cm:  same as fission but with a probed CPU cost model (only when
#                CPU_COST_MODEL is set)
#
# Targets:
#   make all              Build all benchmark executables
#   make bench            Build + time all benchmarks with hyperfine
#   make bench-<name>     Build + time a single benchmark
#   make check            Verify all configs produce identical output
#   make probe            Generate cost model for this machine
#   make clean            Remove build artifacts
#
# Cost model:
#   make probe                              # writes out/costs.json
#   make bench CPU_COST_MODEL=out/costs.json
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

# --- Cost model (empty = use hardcoded defaults) ---
CPU_COST_MODEL ?=

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

# When a cost model file is provided, build a third "fission-cm" config.
ifdef CPU_COST_MODEL
  FISSION_CM_PIPELINE := builtin.module(raise-malloc-to-memref,memory-fission{cpu-cost-model-file=$(abspath $(CPU_COST_MODEL))})
  FISSION_CMS := $(addprefix $(OUT)/,$(addsuffix .fission-cm,$(NAMES)))
else
  FISSION_CMS :=
endif

HYPERFINE_FLAGS ?= --warmup 2 --min-runs 5

# ===== Top-level targets =====================================================

.PHONY: all bench check clean probe

all: $(BASELINES) $(FISSIONS) $(FISSION_CMS)

bench: $(BASELINES) $(FISSIONS) $(FISSION_CMS)
ifdef CPU_COST_MODEL
	@for name in $(NAMES); do \
		echo ""; \
		echo "=== $$name ==="; \
		hyperfine $(HYPERFINE_FLAGS) \
			--command-name "baseline"   "$(OUT)/$$name.baseline" \
			--command-name "fission"    "$(OUT)/$$name.fission" \
			--command-name "fission-cm" "$(OUT)/$$name.fission-cm"; \
	done
else
	@for name in $(NAMES); do \
		echo ""; \
		echo "=== $$name ==="; \
		hyperfine $(HYPERFINE_FLAGS) \
			--command-name "baseline" "$(OUT)/$$name.baseline" \
			--command-name "fission"  "$(OUT)/$$name.fission"; \
	done
endif

check: $(BASELINES) $(FISSIONS) $(FISSION_CMS)
	@fail=0; \
	for name in $(NAMES); do \
		ob=$$($(OUT)/$$name.baseline 2>&1); \
		of=$$($(OUT)/$$name.fission 2>&1); \
		if [ "$$ob" = "$$of" ]; then \
			echo "OK  $$name (baseline vs fission)"; \
		else \
			echo "MISMATCH $$name (baseline vs fission)"; \
			echo "  baseline: $$ob"; \
			echo "  fission:  $$of"; \
			fail=1; \
		fi; \
		if [ -n "$(CPU_COST_MODEL)" ]; then \
			oc=$$($(OUT)/$$name.fission-cm 2>&1); \
			if [ "$$ob" = "$$oc" ]; then \
				echo "OK  $$name (baseline vs fission-cm)"; \
			else \
				echo "MISMATCH $$name (baseline vs fission-cm)"; \
				echo "  baseline:   $$ob"; \
				echo "  fission-cm: $$oc"; \
				fail=1; \
			fi; \
		fi; \
	done; \
	exit $$fail

clean:
	rm -rf $(OUT)

# Generate a cost model for this machine.
probe: | $(OUT)
	$(PROJECT_DIR)/scripts/probe-cost-model.sh $(OUT)/costs.json

# Per-benchmark bench target: make bench-sqrt_consumers
ifdef CPU_COST_MODEL
bench-%: $(OUT)/%.baseline $(OUT)/%.fission $(OUT)/%.fission-cm
	hyperfine $(HYPERFINE_FLAGS) \
		--command-name "baseline"   "$(OUT)/$*.baseline" \
		--command-name "fission"    "$(OUT)/$*.fission" \
		--command-name "fission-cm" "$(OUT)/$*.fission-cm"
else
bench-%: $(OUT)/%.baseline $(OUT)/%.fission
	hyperfine $(HYPERFINE_FLAGS) \
		--command-name "baseline" "$(OUT)/$*.baseline" \
		--command-name "fission"  "$(OUT)/$*.fission"
endif

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

# Step 2b: fission dr-opt passes (hardcoded defaults)
$(OUT)/%.fission.opt.mlir: $(OUT)/%.mlir
	$(DR_OPT) --pass-pipeline='$(FISSION_PIPELINE)' $< -o $@ 2>/dev/null

# Step 2c: fission dr-opt passes (probed cost model)
ifdef CPU_COST_MODEL
$(OUT)/%.fission-cm.opt.mlir: $(OUT)/%.mlir $(CPU_COST_MODEL)
	$(DR_OPT) --pass-pipeline='$(FISSION_CM_PIPELINE)' $< -o $@ 2>/dev/null
endif

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
