// RUN: dr-opt %s -split-input-file -verify-diagnostics

func.func @bad_dims() {
  par.region {
    // expected-error @below {{lowerBounds, upperBounds and steps must have equal length}}
    par.forall([0], [128, 64], [1]) {
    ^bb0(%i: index):
      par.yield
    }
    par.yield
  }
  return
}

// -----

func.func @bad_arg_count() {
  par.region {
    // expected-error @below {{expected 1 index induction-variable block argument(s), got 2}}
    par.forall([0], [128], [1]) {
    ^bb0(%i: index, %j: index):
      par.yield
    }
    par.yield
  }
  return
}
