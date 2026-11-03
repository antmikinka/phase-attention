#!/usr/bin/env bash
# Reproduce the llama.cpp side: clone ggml-org/llama.cpp at the tested base, apply qwen35_phase.patch, CPU build,
# compile the logits dump tool. Local fork only - nothing is pushed anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${LLAMACPP_DIR:-/workspace/llama.cpp-phase}"
BASE=2cbca389f515df455412dafa07e2b1359cc72a6d
CC="${CC:-cc}"; CXX="${CXX:-c++}"     # on the box: /home/box/zigcc/zcc and zc++ (apt mirror was broken, zig via pip)
if [ ! -d "$DIR/.git" ]; then git clone https://github.com/ggml-org/llama.cpp "$DIR"; fi
cd "$DIR"
if ! git rev-parse -q --verify qwen35-phase >/dev/null; then
  git checkout -q "$BASE" && git checkout -q -b qwen35-phase && git -c user.name="${GIT_COMMITTER_NAME:-local}" -c user.email="${GIT_COMMITTER_EMAIL:-local@localhost}" am "$HERE/qwen35_phase.patch"
fi
cmake -S . -B build-cpu -G Ninja -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j "${JOBS:-4}" --target llama
"$CXX" -O2 -std=c++17 -Iinclude -Iggml/include "$HERE/phase_logits.cpp" -Lbuild-cpu/bin -lllama -lggml -lggml-base \
       -Wl,-rpath,"$DIR/build-cpu/bin" -o build-cpu/bin/phase_logits
echo "built $DIR/build-cpu/bin/phase_logits"
