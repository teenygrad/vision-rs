#!/usr/bin/env bash
# Runs `yolo26 validate --runtime runtime` natively on a RISC-V board (e.g. a SpacemiT K3) over ssh.
#
# The board needs nothing installed beyond glibc >= 2.39: this script cross-compiles the example
# for riscv64gc-unknown-linux-gnu, compiles every kernel here (where teenyc lives) by running that
# same binary on one image under qemu-user, then copies the binary, the kernel cache, the weights
# and the first N val2017 images (in validate's filename order) to the board and runs it there
# with teenyc disabled, so a cache miss fails loudly instead of trying to compile.
#
# The board's per-image boxes/scores are copied back to $OUT/board, and the qemu-user run's to
# $OUT/qemu, for an element-wise diff against each other or a CUDA `--dump-outputs` run.
#
# Usage: scripts/riscv-board-validate.sh [-n IMAGES] [-H SSH_HOST] [-o OUT_DIR] [--skip-build]
#   Needs: rustup target riscv64gc-unknown-linux-gnu, gcc-riscv64-linux-gnu, qemu-user, rsync.
#   Env:   source .env first (MODELS_CACHE_DIR, TEENYC_PATH); TEENYC_CAPABILITY (default
#          spacemit-k3); COCO_DIR (default /mnt/data1/datasets/coco-2017); REMOTE_DIR (default
#          vision-rs-validate, relative to the board user's home).
set -euo pipefail

n_images=50
host=jupiter2
out=target/riscv64-native/validate
skip_build=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) n_images="$2"; shift 2 ;;
        -H) host="$2"; shift 2 ;;
        -o) out="$2"; shift 2 ;;
        --skip-build) skip_build=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
: "${MODELS_CACHE_DIR:?source .env first}"
capability=${TEENYC_CAPABILITY:-spacemit-k3}
coco=${COCO_DIR:-/mnt/data1/datasets/coco-2017}
remote=${REMOTE_DIR:-vision-rs-validate}
model=ultralytics/yolo26n
triple=riscv64gc-unknown-linux-gnu
target_dir=target/riscv64-native
bin=$target_dir/$triple/release/examples/yolo26
stage=$target_dir/stage
sysroot=/usr/riscv64-linux-gnu

if [[ $skip_build == 0 ]]; then
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
        CARGO_TARGET_DIR=$target_dir \
        cargo build --release --target $triple --example yolo26 --no-default-features --features riscv
fi

# ── Stage: binary, assets, weights, annotations, the first N images ───────────────────────────
rm -rf "$stage"
mkdir -p "$stage/images" "$stage/models/$model" "$target_dir/kernel-cache" "$out"
out=$(cd "$out" && pwd)
cp "$bin" "$stage/yolo26"
cp -r assets "$stage/assets"
cp "$MODELS_CACHE_DIR/$model/"*.safetensors "$stage/models/$model/"
cp "$coco/annotations/instances_val2017.json" "$stage/instances_val2017.json"
find "$coco/val2017" -maxdepth 1 -name '*.jpg' -printf '%f\n' | LC_ALL=C sort | sed -n "1,${n_images}p" \
    | while read -r f; do cp "$coco/val2017/$f" "$stage/images/"; done

# ── Compile every kernel, by running the binary itself on one image under qemu-user ──────────
# validate compiles the whole graph before its first image, and teenyc (an x86 binary) runs
# natively when qemu-user execs it. Kernels already in the cache aren't recompiled.
rm -rf "$out/qemu"
cache=$PWD/$target_dir/kernel-cache
(
    cd "$stage"
    TEENYC_CACHE_DIR=$cache TEENYC_CAPABILITY=$capability MODELS_CACHE_DIR=$PWD/models \
        qemu-riscv64 -cpu max -L $sysroot ./yolo26 validate --model $model -b 1 --limit 1 \
        --runtime runtime --images images --annotations instances_val2017.json \
        --dump-outputs "$out/qemu"
) | tee "$out/qemu.log"

cp -r "$cache" "$stage/kernel-cache"

# ── Run on the board ──────────────────────────────────────────────────────────────────────────
ssh "$host" "mkdir -p $remote"
rsync -a --delete "$stage/" "$host:$remote/"
ssh "$host" "cd $remote && rm -rf out && uname -a && nproc && \
    TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=\$PWD/kernel-cache \
    TEENYC_CAPABILITY=$capability MODELS_CACHE_DIR=\$PWD/models \
    ./yolo26 validate --model $model -b 1 --limit $n_images --runtime runtime \
        --images images --annotations instances_val2017.json --dump-outputs out" \
    | tee "$out/board.log"
rm -rf "$out/board"
rsync -a "$host:$remote/out/" "$out/board/"
echo "outputs: $out/{qemu,board}  logs: $out/{qemu,board}.log"
