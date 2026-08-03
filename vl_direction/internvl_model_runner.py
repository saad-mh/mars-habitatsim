"""
Loads InternVL and runs single-call image+text generation, intended to run
inside internvl_server.py's process (a future dedicated conda env, mirroring
sam_vla/vlm/qwen_model_runner.py's role for Qwen2.5-VL).

InternVL is not installed anywhere in this repo yet -- no checkpoint has
been chosen and no conda env exists for it. Both functions below are
structural scaffolding that raise NotImplementedError until that happens;
filling them in later should only require touching this one file, since
internvl_server.py only calls these two functions and never imports
transformers/torch itself.
"""

import sys

import numpy as np

INTERNVL_MODEL_ID_PLACEHOLDER = "OpenGVLab/InternVL2-8B"


def load_internvl_model(device: str = "cuda") -> tuple:
    raise NotImplementedError(
        "InternVL is not wired up yet. Once a checkpoint is chosen, this should "
        "mirror qwen_model_runner.load_qwen_model(): "
        "transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True, "
        "torch_dtype='auto', device_map=device) + AutoTokenizer.from_pretrained(...), "
        "run inside a dedicated conda env (see internvl_server_manager.py's "
        "_INTERNVL_VLM_CONDA_ENV) since InternVL's custom modeling code and its "
        "transformers/torch pins may conflict with this repo's other envs."
    )


def run_internvl_inference(
    model,
    tokenizer,
    images: list,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    raise NotImplementedError(
        "InternVL is not wired up yet. Once load_internvl_model() is implemented, "
        "this should follow InternVL's own chat() convention (e.g. "
        "model.chat(tokenizer, pixel_values, prompt, generation_config="
        "{'max_new_tokens': max_new_tokens})) -- see the InternVL repo's README for "
        "the exact multi-image calling convention for the chosen checkpoint."
    )


if __name__ == "__main__":
    try:
        load_internvl_model()
    except NotImplementedError as e:
        print(f"expected: {e}")
