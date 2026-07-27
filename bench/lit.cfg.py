import os
import lit.formats
import lit.util

config.name = "drcompiler-bench"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.drcompiler_obj_root, "bench")

# Corpora without RUN lines: polybench-mlir/ holds raw kernel inputs for the
# benchmark scripts, regpress/ and runtime/ are driven by run_bench.sh (they
# are compiled and timed, not FileCheck'd).
config.excludes = ["polybench-mlir", "regpress", "runtime", "Inputs"]

config.substitutions.append(
    ("%drcompiler_src_root", config.drcompiler_src_root)
)

llvm_tools = [
    "FileCheck",
    "not",
]

tools_dirs = [config.drcompiler_tools_dir, config.llvm_tools_dir]

import lit.llvm

lit.llvm.llvm_config.with_environment("PATH", os.pathsep.join(tools_dirs), append_path=True)
lit.llvm.llvm_config.add_tool_substitutions(["dr-opt"], [config.drcompiler_tools_dir])
lit.llvm.llvm_config.add_tool_substitutions(llvm_tools, [config.llvm_tools_dir])
