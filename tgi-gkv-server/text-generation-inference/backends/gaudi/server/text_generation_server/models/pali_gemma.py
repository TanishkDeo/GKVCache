from io import BytesIO
from PIL import Image
import torch
import torch.distributed
from opentelemetry import trace
from typing import Iterable
from text_generation_server.models.flash_vlm_causal_lm import (
    FlashVlmCausalLMBatch,
    image_text_replacement,
)

from text_generation_server.pb.generate_pb2 import Request

tracer = trace.get_tracer(__name__)


class PaliGemmaBatch(FlashVlmCausalLMBatch):
    @classmethod
    def batch_tokenized_inputs(
        cls, requests: Iterable[Request], tokenizer, processor, config
    ):
        batch_inputs = []
        image_inputs = []
        max_truncation = 0
        for r in requests:
            full_text = ""
            image_id = 0
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    full_text += "<bos>" + chunk.text + "\n"
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    # TODO do_convert_RGB should be on by default ?
                    image = image.convert("RGB")
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(
                        processor, image_input, config, image_id
                    )
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_truncation,
            add_special_tokens=False,
        )["input_ids"]
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            if "pixel_attention_mask" in image_input:
                new_image_inputs["pixel_attention_mask"] = torch.cat(
                    [img["pixel_attention_mask"] for img in image_inputs], dim=0
                )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None
        return batch_tokenized_inputs, image_inputs
