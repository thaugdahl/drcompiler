import os
import lit.formats
import lit.util

config.name = "drcompiler"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.drcompiler_obj_root, "test")

# Tools used in RUN lines.
config.substitutions.append(
    ("%drcompiler_src_root", config.drcompiler_src_root)
)

# Make dr-opt, FileCheck, not, etc. discoverable.
llvm_tools = [
    "FileCheck",
    "not",
]

tools_dirs = [config.drcompiler_tools_dir, config.llvm_tools_dir]

import lit.llvm

lit.llvm.llvm_config.with_environment("PATH", os.pathsep.join(tools_dirs), append_path=True)
lit.llvm.llvm_config.add_tool_substitutions(["dr-opt"], [config.drcompiler_tools_dir])
lit.llvm.llvm_config.add_tool_substitutions(llvm_tools, [config.llvm_tools_dir])
