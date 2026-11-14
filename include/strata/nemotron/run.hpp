// include/strata/nemotron/run.hpp - the nemotron_h_moe engine path's command line (src/nemotron/engine.cu), for `strata`'s family
// switch (src/program/generate.cpp) and strata-nemotron.
#pragma once

namespace strata::nemotron {
/// Runs the engine with strata's command line (--native GGUF --serve ...); returns the exit code.
int run(int argc, char** argv);
}  // namespace strata::nemotron
