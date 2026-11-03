"""Qwen-Drive teacher: load the VLM + BEV head at a chosen precision and expose the two feature taps.

The BEV head reads two tensors from the VLM:
  * ``vit``: pre-merge ViT patch grid after the vision tower's final norm, ``[N_cam, 32, 56, 1024]``
  * ``llm``: language-model hidden states of the image tokens after the final norm, ``[N_cam, 16, 28, 2560]``

``extract_taps`` + ``run_head`` reproduce ``QwenDrivePerception.infer`` but return the taps and raw
head outputs too, which is what distillation and drift measurements need.
"""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from qwen_drive.configuration_qwen_drive import QwenDriveConfig
from qwen_drive_perception import QwenDrivePerception
from qwen_drive_perception.dataset import PerceptionProcessor

PRECISIONS = ("bf16", "int8", "nf4")


class QwenDriveVLM(PreTrainedModel):
    """Only the checkpoint's ``vlm.*`` tensors.

    ``QwenDriveForPlanning`` also builds the ~1B-parameter Planning Expert, which perception
    never uses; skipping it saves memory and loading time.
    """

    config_class = QwenDriveConfig
    base_model_prefix = "vlm"
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: QwenDriveConfig) -> None:
        super().__init__(config)
        self.vlm = AutoModelForImageTextToText.from_config(config.vlm_config)
        self.post_init()


def _offload_device_map(config: QwenDriveConfig, gpu_layers: int) -> dict:
    """bf16 placement for an 8 GB card: vision tower, final norm and the first ``gpu_layers``
    decoder layers on the GPU; embeddings (tied with lm_head) and the remaining layers on CPU."""
    num_layers = config.vlm_config.text_config.num_hidden_layers
    with init_empty_weights():
        model = QwenDriveVLM(config)
    lm = "vlm.model.language_model"
    device_map = {"vlm.model.visual": 0, f"{lm}.embed_tokens": "cpu", "vlm.lm_head": "cpu"}
    for name, _ in model.vlm.model.language_model.named_children():
        if name == "layers":
            for i in range(num_layers):
                device_map[f"{lm}.layers.{i}"] = 0 if i < gpu_layers else "cpu"
        elif name != "embed_tokens":
            device_map[f"{lm}.{name}"] = 0
    return device_map


def load_vlm(weights_dir: str | Path, precision: str, gpu_layers: int = 20, attn_implementation: str = "sdpa"):
    """Load the VLM. ``nf4``/``int8`` quantize the language model only; the vision tower stays bf16
    so the ViT tap is unaffected. ``bf16`` offloads layers past ``gpu_layers`` to CPU."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    weights_dir = Path(weights_dir)
    kwargs = dict(dtype=torch.bfloat16, attn_implementation=attn_implementation)
    if precision == "bf16":
        config = QwenDriveConfig.from_pretrained(weights_dir)
        kwargs["device_map"] = _offload_device_map(config, gpu_layers)
    else:
        # transformers 5.x matches these with re.match from the start of the module path, so bare
        # names like "visual" match nothing; use full prefixes.
        skip = ["vlm.model.visual", "vlm.lm_head", "vlm.model.language_model.embed_tokens"]
        if precision == "nf4":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip,
            )
        else:
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        kwargs.update(quantization_config=quant, device_map={"": 0})
    holder = QwenDriveVLM.from_pretrained(weights_dir, **kwargs)
    return holder.vlm.eval()


def load_head(weights_dir: str | Path, device: str = "cuda") -> QwenDrivePerception:
    head = QwenDrivePerception.from_pretrained(Path(weights_dir) / "perception", dtype=torch.bfloat16)
    return head.to(device).eval()


def load_processor(weights_dir: str | Path) -> PerceptionProcessor:
    return PerceptionProcessor(AutoTokenizer.from_pretrained(weights_dir))


@torch.no_grad()
def extract_taps(vlm, inputs: dict, num_cams: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(vit, llm)`` taps for the last ``num_cams`` images, matching upstream ``infer``.

    Calls ``vlm.model`` rather than ``vlm`` so the 248k-vocab lm_head is never evaluated
    (~1.4 GB of logits for ~2.7k tokens that perception does not use).
    """
    input_ids = inputs["input_ids"]
    image_grid_thw = inputs["image_grid_thw"]
    image_token_id = vlm.config.image_token_id
    visual = vlm.model.visual
    captured = {}
    handle = visual.merger.register_forward_hook(lambda m, args, out: captured.__setitem__("patches", args[0]))
    try:
        outputs = vlm.model(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=(input_ids == image_token_id).long(),
            use_cache=False,
            output_hidden_states=True,
        )
    finally:
        handle.remove()

    # Same post-processing as QwenDrivePerception.infer: final norm on the last hidden state,
    # and the merger's norm on the pre-merge patches.
    hidden = vlm.model.language_model.norm(outputs.hidden_states[-1])
    patches = visual.merger.norm(captured["patches"])
    vit_grids = QwenDrivePerception._premerge_grids(patches, image_grid_thw)

    grid_h, grid_w = image_grid_thw[-1, 1].item(), image_grid_thw[-1, 2].item()
    tokens_per_img = grid_h // 2 * grid_w // 2
    llm_tokens = hidden[0][input_ids[0] == image_token_id][-num_cams * tokens_per_img :]
    llm = llm_tokens.view(num_cams, grid_h // 2, grid_w // 2, -1)
    vit = torch.stack(vit_grids[-num_cams:], dim=0)
    return vit, llm


@torch.no_grad()
def run_head(head: QwenDrivePerception, vit: torch.Tensor, llm: torch.Tensor, img_metas: dict) -> dict:
    """Run the BEV head on taps. Returns decoded predictions plus raw logits for distillation."""
    bev = head.bev_modeling
    device = next(bev.parameters()).device
    outs = bev(
        img_vit_feats=vit.to(device, torch.bfloat16),
        img_llm_feats=llm.to(device, torch.bfloat16),
        img_metas=[img_metas],
    )
    det = bev.head.get_bboxes(outs, [img_metas])[0]
    return {
        "boxes": det["boxes"],
        "scores": det["scores"],
        "labels": det["labels"],
        "occ": bev.head.get_occ(outs, [img_metas])[0].byte().cpu().numpy(),
        "map": bev.head.get_map_seg(outs, [img_metas])[0].byte().cpu().numpy(),
        "det_cls_logits": outs["all_cls_scores"][-1][0].float().cpu().numpy(),
        "det_box_preds": outs["all_bbox_preds"][-1][0].float().cpu().numpy(),
        "occ_logits": outs["occ_pred"][0].to(torch.float16).cpu().numpy(),
        "map_logits": outs["seg_preds"][0].to(torch.float16).cpu().numpy(),
    }
