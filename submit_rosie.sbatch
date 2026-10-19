#!/bin/bash
# ==============================================================================
# Rockwell Automation + NVIDIA Innovation Lab: MSOE Rosie Slurm Batch Script
# ==============================================================================
# Strict GPU & RTX Ray-Tracing Slurm Batch Script for MSOE Rosie Supercomputer.
#
# Constraints Enforced:
#   1. STRICT GPU ENFORCEMENT: Fails immediately if an NVIDIA GPU is not allocated.
#   2. NO CPU FALLBACK: Strictly runs NVIDIA Isaac Sim Omniverse Replicator with
#      hardware RTX ray-tracing. Does not fallback to CPU/2D image generators.
#   3. GLIBC 2.38 WORKAROUND: Injects container-native libGLX and prioritizes
#      NVIDIA Container CLI (--nvccli / --nv + LD_PRELOAD) to bypass the host
#      Ubuntu 24.04 graphics library collision.
#   4. PERSISTENT RTX SHADER CACHES: Mounts kit_cache, glcache, and computecache
#      to prevent cold-start shader compile timeouts.
#   5. Multi-modal packaging: Output dataset compressed to .tar.gz for download.
# ==============================================================================

#SBATCH --job-name=csi_rtx_sdg
#SBATCH --output=rosie_rtx_%j.log
#SBATCH --error=rosie_rtx_%j.err
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

echo "======================================================================"
echo "  MSOE Rosie: NVIDIA Isaac Sim RTX Raytraced Synthetic Pipeline        "
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Compute Node: $SLURMD_NODENAME"
echo "  Submit Dir:   $SLURM_SUBMIT_DIR"
echo "  Start Time:   $(date)"
echo "======================================================================"

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------------------
# 1. Enforce NVIDIA GPU Hardware & Drivers
# ------------------------------------------------------------------------------
echo -e "\n[1/5] Enforcing GPU Allocation on Compute Node..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[FATAL ERROR] nvidia-smi not found. This job requires an NVIDIA GPU node."
    echo "Check your Slurm allocation or partition flags."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)

echo "  ✓ Physical GPU Detected: $GPU_NAME"
echo "  ✓ NVIDIA Driver Version: $DRIVER_VER"
echo "  ✓ Total GPU Memory:     $TOTAL_MEM"

# Initialize Environment Modules in batch subshell
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
elif [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
fi

# ------------------------------------------------------------------------------
# 2. Container Runtime (Apptainer / Singularity) Setup
# ------------------------------------------------------------------------------
echo -e "\n[2/5] Initializing Container Runtime..."
CONTAINER_CMD=""
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
elif command -v module &> /dev/null; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
    fi
fi

if [ -z "$CONTAINER_CMD" ]; then
    echo "[FATAL ERROR] Neither apptainer nor singularity found on this node."
    echo "Ensure 'module load apptainer' or 'module load singularity' is available."
    exit 1
fi
echo "  ✓ Using Container Runtime: $CONTAINER_CMD ($($CONTAINER_CMD --version))"

# Prevent filling node /tmp RAMdisk
export APPTAINER_CACHEDIR="${HOME}/.cache/apptainer"
export APPTAINER_TMPDIR="${PWD}/.apptainer_tmp"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ------------------------------------------------------------------------------
# 3. Locate or Pull Official NVIDIA Isaac Sim SIF Image
# ------------------------------------------------------------------------------
echo -e "\n[3/5] Locating NVIDIA Isaac Sim Container Image (.sif)..."

SIF_CANDIDATES=(
    "${ISAAC_SIF_PATH:-}"
    "$PWD/isaac-sim.sif"
    "$HOME/nvidia_hackathon/isaac-sim.sif"
    "$HOME/containers/isaac-sim.sif"
    "$HOME/containers/isaac-sim_4.2.0.sif"
    "$HOME/isaac-sim.sif"
    "/data/containers/isaac-sim_4.2.0.sif"
    "/data/containers/isaac-sim-6.1.0.sif"
    "/data/containers/isaac-sim_5.0.0.sif"
    "/data/containers/isaac-sim.sif"
    "/data/containers/isaac-sim-20231103.sif"
    "/shared/containers/isaac-sim.sif"
)

SIF_IMAGE=""
for candidate in "${SIF_CANDIDATES[@]}"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        SIF_IMAGE="$candidate"
        break
    fi
done

if [ -z "$SIF_IMAGE" ]; then
    echo "[!] No existing Isaac Sim .sif found in standard cluster locations."
    echo "    Attempting to pull and build official NVIDIA Isaac Sim 4.2.0 from NGC..."
    mkdir -p "$HOME/containers"
    SIF_IMAGE="$HOME/containers/isaac-sim_4.2.0.sif"
    
    $CONTAINER_CMD build "$SIF_IMAGE" docker://nvcr.io/nvidia/isaac-sim:4.2.0
fi

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[FATAL ERROR] Isaac Sim container image could not be loaded: $SIF_IMAGE"
    echo "Please build or copy the container SIF image before running."
    exit 1
fi
echo "  ✓ Isaac Sim Container SIF: $SIF_IMAGE"

# ------------------------------------------------------------------------------
# 4. Configure GPU Driver & OpenUSD Library Passthrough
# ------------------------------------------------------------------------------
echo -e "\n[4/5] Activating Container GPU Passthrough & OpenUSD Libraries..."

GPU_RUN_FLAG="--nv"

# Explicitly append internal Pixar OpenUSD runtime binaries to library search path
USD_BIN_PATHS="/isaac-sim/extscache/omni.usd.libs-1.0.1+10a4b5c0.lx64.r.cp310/bin:/isaac-sim/extscache/omni.usd-1.12.2+10a4b5c0.lx64.r.cp310/bin"
export SINGULARITYENV_APPEND_LD_LIBRARY_PATH="$USD_BIN_PATHS"
export APPTAINERENV_APPEND_LD_LIBRARY_PATH="$USD_BIN_PATHS"

# Extract container-native Ubuntu 22.04 GLVND libraries to mask host GLIBC 2.38 collision
COMPAT_GL_DIR="$HOME/.cache/rosie_compat_gl"
mkdir -p "$COMPAT_GL_DIR"
if [ ! -f "$COMPAT_GL_DIR/libGLdispatch.so.0" ] || [ ! -f "$COMPAT_GL_DIR/libGLX.so.0" ]; then
    echo "  Extracting container-native GLVND libraries to bypass host GLIBC 2.38 clash..."
    $CONTAINER_CMD exec "$SIF_IMAGE" cp -L /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 "$COMPAT_GL_DIR/" 2>/dev/null || true
    $CONTAINER_CMD exec "$SIF_IMAGE" cp -L /usr/lib/x86_64-linux-gnu/libGLX.so.0 "$COMPAT_GL_DIR/" 2>/dev/null || true
fi

echo "  ✓ Container GPU Passthrough: $GPU_RUN_FLAG"
echo "  ✓ Appended OpenUSD Path:    $USD_BIN_PATHS"
echo "  ✓ GLIBC 2.38 Masking Path:  $COMPAT_GL_DIR"

# ------------------------------------------------------------------------------
# 5. Launch Headless RTX Raytraced Generation
# ------------------------------------------------------------------------------
echo -e "\n[5/5] Launching Omniverse Replicator with RTX Ray Tracing..."

CACHE_BASE="$HOME/.cache/rosie_isaac_sim"
mkdir -p "$CACHE_BASE/computecache" \
         "$CACHE_BASE/logs" \
         "$CACHE_BASE/kit_cache"

OUTPUT_DIR="./rosie_synthetic_dataset"
mkdir -p "$OUTPUT_DIR"
NUM_REPEATS=5

USD_STAGE="${CSI_USD_PATH:-}"
USD_ARG=""
if [ -n "$USD_STAGE" ] && [ -f "$USD_STAGE" ]; then
    USD_ARG="--usd-stage $USD_STAGE"
fi

echo "  Target Output:  $OUTPUT_DIR"
echo "  Batch Repeats:  $NUM_REPEATS"
echo "  RTX Driver Bypass Flag: --/rtx/verifyDriverVersion/enabled=false"
echo "  Multi-GPU Split Flag:   --/renderer/multiGpu/enabled=false"
echo "  Renderer Active GPU:    --/renderer/activeGpu=0"
echo "----------------------------------------------------------------------"

export SINGULARITYENV_PYTHONUNBUFFERED="1"
export APPTAINERENV_PYTHONUNBUFFERED="1"

START_TIME=$(date +%s)

$CONTAINER_CMD exec $GPU_RUN_FLAG --writable-tmpfs \
    --env "ACCEPT_EULA=Y" \
    --env "PRIVACY_CONSENT=Y" \
    --env "OMNI_KIT_ALLOW_UNSUPPORTED_DRIVER=1" \
    --env "PYTHONUNBUFFERED=1" \
    --env "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0" \
    --env "APPEND_LD_LIBRARY_PATH=$USD_BIN_PATHS" \
    --env "LD_LIBRARY_PATH=$USD_BIN_PATHS:\$LD_LIBRARY_PATH" \
    --bind "$COMPAT_GL_DIR/libGLdispatch.so.0:/.singularity.d/libs/libGLdispatch.so.0" \
    --bind "$COMPAT_GL_DIR/libGLX.so.0:/.singularity.d/libs/libGLX.so.0" \
    --bind "$CACHE_BASE/kit_cache:/isaac-sim/kit/cache" \
    --bind "$CACHE_BASE/computecache:/isaac-sim/.nv/ComputeCache" \
    --bind "$CACHE_BASE/logs:/isaac-sim/kit/logs" \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    "$SIF_IMAGE" \
    /isaac-sim/python.sh synthetic_pipeline/isaac_sim_replicator_generator.py \
        $USD_ARG \
        --config synthetic_pipeline/config.yaml \
        --output-dir "$OUTPUT_DIR" \
        --num-repeats "$NUM_REPEATS" \
        --headless \
        --/rtx/verifyDriverVersion/enabled=false \
        --/renderer/multiGpu/enabled=false \
        --/renderer/activeGpu=0

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ------------------------------------------------------------------------------
# 6. Packaging & Validation
# ------------------------------------------------------------------------------
echo -e "\n======================================================================"
echo "  ROSIE RTX RAYTRACED SDG COMPLETE IN ${DURATION}s!                   "
echo "======================================================================"

# Ingest any frames from Replicator BasicWriter fallback
if [ -d "${OUTPUT_DIR}/replicator_raw" ]; then
    find "${OUTPUT_DIR}/replicator_raw" -type f \( -name "*.png" -o -name "*.jpg" \) | while read -r raw_file; do
        bname=$(basename "$raw_file")
        if [ ! -f "${OUTPUT_DIR}/images/$bname" ]; then
            cp -n "$raw_file" "${OUTPUT_DIR}/images/" 2>/dev/null || true
        fi
    done
fi

IMG_COUNT=$(find "${OUTPUT_DIR}/images" -type f 2>/dev/null | wc -l || echo "0")
LBL_COUNT=$(find "${OUTPUT_DIR}/labels_yolo" -type f 2>/dev/null | wc -l || echo "0")
META_COUNT=$(find "${OUTPUT_DIR}/metadata" -type f 2>/dev/null | wc -l || echo "0")

echo "  • Rendered RTX Images: $IMG_COUNT"
echo "  • YOLOv8 Labels:       $LBL_COUNT"
echo "  • Metadata JSONs:      $META_COUNT"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "[ERROR] No images were rendered during the execution."
    exit 1
fi

ARCHIVE_NAME="rosie_csi_rtx_dataset_$(date +%Y%m%d_%H%M%S).tar.gz"
echo -e "\nCompressing RTX dataset into archive: $ARCHIVE_NAME..."
tar -czf "$ARCHIVE_NAME" "$OUTPUT_DIR"
ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | awk '{print $1}')

echo "  ✓ Archive successfully created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
echo -e "\n======================================================================"
echo "  SUCCESS: JOB FINISHED ON GPU ($GPU_NAME)"
echo "======================================================================"
