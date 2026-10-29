#!/usr/bin/env bash
# run-in-container.sh OUT FAMILIES CHANGEOVERS SHARD: render_carryover.sh inside the Isaac Sim 6.0.1
# container of the Brev Isaac launchable, with the repo setup/ and twin assets under ~/linewatch.
# Use the nvidia runtime: "--gpus all" drops the graphics capability and the path tracer hangs.
# Isaac caches persist in ~/isaac-cache so only the first launch compiles shaders.
# FIXTURE_ARGS, if set, is passed to the fixture builder (see render_carryover.sh).
out="$1"; shift
mkdir -p /home/ubuntu/linewatch/$out /home/ubuntu/isaac-cache/{kit,ov,pip,glcache,compute,logs,data}
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --network host -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e FIXTURE_ARGS="${FIXTURE_ARGS:-}" \
  -v /home/ubuntu/linewatch:/work -v /home/ubuntu/isaac-cache/kit:/isaac-sim/kit/cache:rw \
  -v /home/ubuntu/isaac-cache/ov:/root/.cache/ov:rw -v /home/ubuntu/isaac-cache/pip:/root/.cache/pip:rw \
  -v /home/ubuntu/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw -v /home/ubuntu/isaac-cache/compute:/root/.nv/ComputeCache:rw \
  -v /home/ubuntu/isaac-cache/logs:/root/.nvidia-omniverse/logs:rw -v /home/ubuntu/isaac-cache/data:/root/.local/share/ov/data:rw \
  --user root --entrypoint bash nvcr.io/nvidia/isaac-sim:6.0.1 \
  -c "cd /work && bash setup/brev/render_carryover.sh /work/$out /work/datasets/mixing_twin/assets/csi_digital_twin_v1 $*" \
  > /home/ubuntu/linewatch/$out.console.log 2>&1
echo "exit $?" >> /home/ubuntu/linewatch/$out.console.log
