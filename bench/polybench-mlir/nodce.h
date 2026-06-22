#include "polybench.h"
#undef polybench_prevent_dce
#define polybench_prevent_dce(func) do {} while(0)
