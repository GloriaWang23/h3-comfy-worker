# H3 Studio worker: the official runpod/worker-comfyui base image, extended the way its
# customization guide recommends (FROM <version>-base), with what MiniMax H3 needs on top:
#  - torch 2.11.0+cu130: comfy-kitchen's CUDA int8 kernels (used by the int8_convrot
#    checkpoints) are disabled below CUDA 13. The endpoint only schedules hosts with
#    CUDA >= 13.0, so the driver requirement of cu13 wheels is always met.
#  - ComfyUI ${COMFYUI_VERSION}: the base ships 0.34.0, which has no MiniMax H3 nodes.
#  - h3_handler.py: media URLs in, Tencent COS URLs out, cancel -> ComfyUI interrupt.
# Model weights are NOT baked in: they come from Runpod's cached-model feature (the
# Hugging Face repo GloriaWang23/h3-comfy-runtime), linked to /h3-models at start.
FROM runpod/worker-comfyui:5.10.0-base

ARG COMFYUI_VERSION=v0.39.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

# 1. torch cu130 in the launch venv (/opt/venv), replacing the base image's cu128 build.
RUN uv pip install --reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url ${TORCH_INDEX_URL} \
    && rm -rf /root/.cache

# 2. Replace ComfyUI with the pinned release and mirror its requirements into /opt/venv.
#    torch is already satisfied, so the bare `torch` requirement leaves it alone. The
#    transformers / huggingface-hub caps are the same ones the base image applies.
RUN cp /comfyui/extra_model_paths.yaml /tmp/extra_model_paths.yaml \
    && rm -rf /comfyui \
    && git clone --depth 1 --branch ${COMFYUI_VERSION} https://github.com/Comfy-Org/ComfyUI.git /comfyui \
    && rm -rf /comfyui/.git \
    && mv /tmp/extra_model_paths.yaml /comfyui/extra_model_paths.yaml \
    && uv pip install -r /comfyui/requirements.txt "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && rm -rf /root/.cache

# 3. Model folders from the cached Hugging Face snapshot (see h3_start.sh).
RUN printf '\nh3_hf_cache:\n  base_path: /h3-models\n  diffusion_models: diffusion_models/\n  text_encoders: text_encoders/\n  vae: vae/\n  loras: loras/\n' \
      >> /comfyui/extra_model_paths.yaml

# Build-time smoke test, as in the base image: start ComfyUI on CPU so an import or
# dependency break fails the build instead of a live worker.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# 4. Handler and start wrapper.
COPY h3_handler.py /h3_handler.py
COPY h3_start.sh /h3_start.sh
RUN chmod +x /h3_start.sh \
    && sed -i 's#python -u /handler.py#python -u /h3_handler.py#' /start.sh \
    && grep -q "python -u /h3_handler.py" /start.sh \
    && echo "${COMFYUI_VERSION}" > /comfyui/.h3_version

CMD ["/h3_start.sh"]
