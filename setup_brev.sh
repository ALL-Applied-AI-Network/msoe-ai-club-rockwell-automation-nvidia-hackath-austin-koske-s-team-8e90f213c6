#!/usr/bin/env bash
# ==============================================================================
# Rockwell Automation + NVIDIA Innovation Lab: Brev.dev Turnkey SDG Pipeline
# ==============================================================================
# Automates physical AI synthetic data generation on NVIDIA Brev.dev using Isaac Sim.
# Features:
#   1. Automatic detection of Bare-Metal Isaac Sim vs Official NGC Docker Container
#   2. Automatic shader cache volume mounting to prevent repeated recompilations
#   3. Omniverse Replicator headless raytracing with procedural digital twin fallback
#   4. Mass-conserving subtractive Beer-Lambert fluid optical absorption
#   5. Multi-modal export (YOLOv8, Digital Twin JSON, Cosmos Reason VLM, CSV Manifest)
#   6. Automated dataset compression and 1-click 'brev scp' local download commands
#
# Usage:
#   bash synthetic_pipeline/setup_brev.sh [OPTIONS]
#
# Options:
#   --usd-stage PATH         Path to CSI Manufacturing Testbed USD file (optional)
#   --num-repeats N          Number of repetitions per recipe transition (default: 5)
#   --output-dir DIR         Output directory for synthetic dataset (default: ./synthetic_brev_dataset)
#   --container-image IMG    Docker container tag (default: nvcr.io/nvidia/isaac-sim:4.2.0)
#   --force-docker           Force execution via Docker even if host python.sh exists
#   --test-only              Fast sanity check (1 repeat, ~36 samples)
#   --help                   Display this help message
# ==============================================================================

set -euo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "  Rockwell + NVIDIA Innovation Lab: Isaac Sim Brev.dev Turnkey SDG    "
echo "======================================================================"
echo -e "${RESET}"

# Default parameters
USD_STAGE="${CSI_USD_PATH:-}"
NUM_REPEATS=5
OUTPUT_DIR="./synthetic_brev_dataset"
CONFIG_PATH="synthetic_pipeline/config.yaml"
CONTAINER_IMAGE="${ISAAC_CONTAINER_IMAGE:-nvcr.io/nvidia/isaac-sim:4.2.0}"
FORCE_DOCKER=false

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --usd-stage)
      USD_STAGE="$2"
      shift 2
      ;;
    --num-repeats)
      NUM_REPEATS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --container-image)
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    --force-docker)
      FORCE_DOCKER=true
      shift
      ;;
    --test-only)
      NUM_REPEATS=1
      shift
      ;;
    --help)
      echo "Usage: bash synthetic_pipeline/setup_brev.sh [OPTIONS]"
      echo "  --usd-stage PATH         Path to CSI digital twin USD stage (optional)"
      echo "  --num-repeats N          Number of repeats per transition (default: 5)"
      echo "  --output-dir DIR         Output directory (default: ./synthetic_brev_dataset)"
      echo "  --container-image IMG    Isaac Sim NGC image (default: nvcr.io/nvidia/isaac-sim:4.2.0)"
      echo "  --force-docker           Run inside Docker container even if native Isaac Sim exists"
      echo "  --test-only              Run single repetition sanity check (36 samples)"
      exit 0
      ;;
    *)
      echo -e "${RED}[!] Unknown argument: $1${RESET}"
      exit 1
      ;;
  esac
done

# Step 1: Verify NVIDIA GPU & Driver Environment
echo -e "${BOLD}[Step 1/5] Verifying GPU Environment...${RESET}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[ERROR] nvidia-smi not found. This script must be run on an NVIDIA GPU instance.${RESET}"
    echo "Tip: On Brev.dev, select an instance with an NVIDIA L40S, A100, A10G, or RTX 4090 GPU."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | awk '{print $3}' || echo "N/A")
echo -e "  ${GREEN}✓ Detected GPU:${RESET}    ${BOLD}${GPU_NAME}${RESET}"
echo -e "  ${GREEN}✓ NVIDIA Driver:${RESET}   ${DRIVER_VER}"
echo -e "  ${GREEN}✓ CUDA Version:${RESET}    ${CUDA_VER}"

# Step 2: Determine Execution Strategy (Bare-Metal vs NGC Docker Container)
echo -e "\n${BOLD}[Step 2/5] Determining Isaac Sim Runtime Environment...${RESET}"

ISAAC_PATH=""
if [ "${FORCE_DOCKER}" = false ]; then
    POSSIBLE_PATHS=(
        "${ISAAC_PATH_ENV:-}"
        "/isaac-sim"
        "/workspace/isaac-sim"
        "$HOME/.local/share/ov/pkg/isaac-sim-*"
        "/root/.local/share/ov/pkg/isaac-sim-*"
        "/opt/nvidia/omniverse/isaac-sim"
    )

    for p in "${POSSIBLE_PATHS[@]}"; do
        for expanded in $p; do
            if [ -f "${expanded}/python.sh" ]; then
                ISAAC_PATH="${expanded}"
                break 2
            fi
        done
    done

    if [ -z "${ISAAC_PATH}" ]; then
        FOUND_PATH=$(find / -name "python.sh" 2>/dev/null | grep "isaac-sim" | head -n 1 || true)
        if [ -n "${FOUND_PATH}" ]; then
            ISAAC_PATH=$(dirname "${FOUND_PATH}")
        fi
    fi
fi

# Step 3: Validate USD Stage / Procedural Scene
echo -e "\n${BOLD}[Step 3/5] Validating Digital Twin Stage...${RESET}"
USD_ARG=""
if [ -n "${USD_STAGE}" ]; then
    if [ ! -f "${USD_STAGE}" ]; then
        echo -e "${RED}[ERROR] Specified USD stage not found: ${USD_STAGE}${RESET}"
        exit 1
    fi
    echo -e "  ${GREEN}✓ Using USD stage:${RESET} ${USD_STAGE}"
    USD_ARG="--usd-stage ${USD_STAGE}"
else
    # Auto-detect existing USD files if present in workspace
    CANDIDATES=$(find . -maxdepth 3 -name "*.usd" -o -name "*.usda" -o -name "*.usdc" 2>/dev/null | grep -v "mock_testbed" || true)
    if [ -n "${CANDIDATES}" ]; then
        DETECTED_USD=$(echo "${CANDIDATES}" | head -n 1)
        echo -e "  ${YELLOW}* No USD stage specified. Auto-detected:${RESET} ${DETECTED_USD}"
        USD_ARG="--usd-stage ${DETECTED_USD}"
    else
        echo -e "  ${CYAN}ℹ No CAD USD stage supplied. Running Procedural Digital Twin in-memory.${RESET}"
        USD_ARG=""
    fi
fi

# Step 4: Execute Generation
echo -e "\n${BOLD}[Step 4/5] Launching Omniverse Replicator Generation Pipeline...${RESET}"
mkdir -p "${OUTPUT_DIR}"
START_TIME=$(date +%s)

if [ -n "${ISAAC_PATH}" ] && [ "${FORCE_DOCKER}" = false ]; then
    # Strategy A: Native Host / Pre-installed Bare-Metal
    PYTHON_EXEC="${ISAAC_PATH}/python.sh"
    echo -e "  ${GREEN}✓ Runtime Mode:${RESET}   Bare-Metal Native (${ISAAC_PATH})"
    echo -e "  ${GREEN}✓ Launcher:${RESET}       ${PYTHON_EXEC}"

    echo "  Verifying Python dependencies..."
    ${PYTHON_EXEC} -m pip install --quiet pyyaml scipy pillow pandas || true

    echo "----------------------------------------------------------------------"
    ${PYTHON_EXEC} synthetic_pipeline/isaac_sim_replicator_generator.py \
        ${USD_ARG} \
        --config "${CONFIG_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-repeats "${NUM_REPEATS}" \
        --headless
else
    # Strategy B: Turnkey NVIDIA NGC Docker Container
    echo -e "  ${YELLOW}* Bare-metal Isaac Sim not detected. Activating Turnkey Docker Runtime!${RESET}"
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Neither bare-metal Isaac Sim nor Docker was found.${RESET}"
        echo "Please install Docker and NVIDIA Container Toolkit or launch an official Brev Isaac Sim instance."
        exit 1
    fi

    echo -e "  ${GREEN}✓ Runtime Mode:${RESET}   NVIDIA NGC Docker (${CONTAINER_IMAGE})"
    echo -e "  ${GREEN}✓ EULA:${RESET}           Accepted (ACCEPT_EULA=Y, PRIVACY_CONSENT=Y)"

    # Configure persistent shader and Omniverse cache directories
    CACHE_BASE="${HOME}/docker/isaac-sim/cache"
    mkdir -p "${CACHE_BASE}/kit" "${CACHE_BASE}/ov" "${CACHE_BASE}/pip" \
             "${CACHE_BASE}/glcache" "${CACHE_BASE}/computecache" \
             "${HOME}/docker/isaac-sim/logs"

    echo "  Mounting persistent GPU shader caches to prevent cold-start recompilation..."
    echo "  Starting containerized Replicator execution..."
    echo "----------------------------------------------------------------------"

    docker run --name isaac-sim-sdg-brev \
        --entrypoint bash \
        --runtime=nvidia \
        --gpus all \
        --rm \
        --network=host \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        -v "${CACHE_BASE}/kit:/isaac-sim/kit/cache:rw" \
        -v "${CACHE_BASE}/ov:/root/.cache/ov:rw" \
        -v "${CACHE_BASE}/pip:/root/.cache/pip:rw" \
        -v "${CACHE_BASE}/glcache:/root/.cache/nvidia/GLCache:rw" \
        -v "${CACHE_BASE}/computecache:/root/.nv/ComputeCache:rw" \
        -v "${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
        -v "$(pwd):/workspace/project:rw" \
        -w /workspace/project \
        "${CONTAINER_IMAGE}" \
        -c "/isaac-sim/python.sh -m pip install --quiet pyyaml scipy pillow pandas && \
            /isaac-sim/python.sh synthetic_pipeline/isaac_sim_replicator_generator.py \
            ${USD_ARG} \
            --config ${CONFIG_PATH} \
            --output-dir ${OUTPUT_DIR} \
            --num-repeats ${NUM_REPEATS} \
            --headless"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Step 5: Packaging & Summary
echo -e "\n${BOLD}[Step 5/5] Finalizing Dataset & Artifact Packaging...${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
echo -e "${GREEN}${BOLD}  SYNTHETIC DATA GENERATION COMPLETED IN ${DURATION}s!                 ${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"

IMG_COUNT=$(find "${OUTPUT_DIR}/images" -type f 2>/dev/null | wc -l || echo "0")
LBL_COUNT=$(find "${OUTPUT_DIR}/labels_yolo" -type f 2>/dev/null | wc -l || echo "0")
META_COUNT=$(find "${OUTPUT_DIR}/metadata" -type f 2>/dev/null | wc -l || echo "0")

echo -e "  • Total Images Rendered:       ${BOLD}${IMG_COUNT}${RESET}"
echo -e "  • YOLO Annotation Files:       ${BOLD}${LBL_COUNT}${RESET}"
echo -e "  • Digital Twin Metadata JSONs: ${BOLD}${META_COUNT}${RESET}"
echo -e "  • Cosmos Reason Prompts:       ${BOLD}${OUTPUT_DIR}/cosmos_reason_prompts.jsonl${RESET}"
echo -e "  • Master Dataset Manifest:     ${BOLD}${OUTPUT_DIR}/dataset_manifest.csv${RESET}"

# Package artifact for download
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="synthetic_csi_dataset_${TIMESTAMP}.tar.gz"
echo -e "\n${BOLD}[Packaging] Compressing dataset into ${ARCHIVE_NAME}...${RESET}"
chmod -R 777 "${OUTPUT_DIR}" 2>/dev/null || sudo chmod -R 777 "${OUTPUT_DIR}" 2>/dev/null || true
tar -czf "${ARCHIVE_NAME}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
ARCHIVE_SIZE=$(du -h "${ARCHIVE_NAME}" | awk '{print $1}')
echo -e "${GREEN}✓ Archive created successfully: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})${RESET}"

INSTANCE_NAME="${BREV_INSTANCE_NAME:-<instance-name>}"
echo -e "\n${CYAN}${BOLD}=== Turnkey Download Instructions ===${RESET}"
echo "Run this command on your local computer to download the dataset:"
echo -e "  ${BOLD}brev scp ${INSTANCE_NAME}:${PWD}/${ARCHIVE_NAME} ./downloads/${RESET}"
echo -e "\n${YELLOW}${BOLD}=== Cost Optimization Reminder ===${RESET}"
echo "To stop billing while keeping your data saved on Brev:"
echo -e "  ${BOLD}brev stop ${INSTANCE_NAME}${RESET}"
echo "Or if completely finished with the instance:"
echo -e "  ${BOLD}brev delete ${INSTANCE_NAME}${RESET}"
