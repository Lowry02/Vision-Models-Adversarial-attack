from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

class QwenChat():
  def __init__(self):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    token = "" # your token
    
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_id,
      torch_dtype=torch.float16,
      device_map="auto",
      token=token
    ).eval()
    self.processor = AutoProcessor.from_pretrained(model_id, token=token)
    self.device = self.model.device
    self.vocab_size = self.model.config.vocab_size

  def ask(self, messages, max_new_tokens=10):
    with torch.no_grad():
      texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
      ]

      image_inputs, video_inputs = process_vision_info(messages)
      inputs = self.processor(
          text=texts,
          images=image_inputs,
          videos=video_inputs,
          padding=True,
          padding_side='left',
          return_tensors="pt",
      )
      inputs = inputs.to("cuda")

      generated_ids = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        use_model_defaults=True,
        return_dict_in_generate=True,
        output_scores=True,
      )

      generated_ids_trimmed = [
          out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids['sequences'])
      ]
      output_text = self.processor.batch_decode(
          generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
      )
      return output_text, generated_ids['scores'][0]