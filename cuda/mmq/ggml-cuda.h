// SPDX-License-Identifier: MIT
// Thin redirect so the vendored mmq/vecdotq/common code can write
// #include "ggml-cuda.h" unchanged. Ported from q38-main (COPY -> RENAME -> MOVE).
#pragma once
#include "q3_ggml_stubs.h"
