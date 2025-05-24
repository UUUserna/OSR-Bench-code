import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载Qwen2.5-VL模型和处理器"""
    print(f"正在加载模型: {model_path}")
    image_pixels = 1024 * 512  # 图片分辨率

    # 设置适当的像素范围
    min_pixels = 256 * 28 * 28  # 最小像素阈值
    max_pixels = image_pixels  # 最大像素设为图片的实际像素数
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 如果不使用Flash Attention，设置环境变量
    if not use_flash_attention:
        os.environ["USE_FLASH_ATTENTION"] = "0"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
    else:
        # 使用Flash Attention 2进行内存优化
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  
            attn_implementation="flash_attention_2",  
            device_map="auto"  
        )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    
    print(f"模型加载完成")
    return model, processor

def generate(model, processor, messages, device='cuda'):
    """使用Qwen2.5-VL模型生成回答"""
    # 准备输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 使用混合精度进行推理以减少内存使用
    with torch.cuda.amp.autocast():
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移动到指定设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 释放CPU内存
        del image_inputs, video_inputs
        
        # 使用内存高效的设置进行生成
        with torch.no_grad():  
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512
            )
        
        # 处理输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return output_text[0]

def prepare_message(image_path, question, prev_qa=None):

    image = Image.open(image_path)
    
    """准备Qwen模型的消息格式"""
    if prev_qa:
        # 多轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prev_qa["question"]},
                ],
            },
            {"role": "assistant", "content": prev_qa["answer"]},
            {"role": "user", "content": question},
        ]
    else:
        # 单轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
    
    return messages