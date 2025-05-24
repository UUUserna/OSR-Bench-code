import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载LLaVA模型和处理器"""
    print(f"正在加载LLaVA模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 加载处理器
    processor = LlavaNextProcessor.from_pretrained(model_path)
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    print(f"LLaVA模型加载完成")
    return model, processor

def generate(model, processor, messages, device='cuda'):
    """使用LLaVA模型生成回答"""
    # 提取图像
    image = None
    for msg in messages:
        if msg["role"] == "user":
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image")
                    break
            if image:
                break
    
    if image is None:
        raise ValueError("未找到图像输入")
    
    # 使用processor的应用聊天模板功能格式化对话
    prompt = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    
    # 使用混合精度进行推理以减少内存使用
    with torch.cuda.amp.autocast():
        # 处理输入
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 解码输出
        generated_text = processor.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        # 提取模型生成的部分（去除原始提示）
        response = generated_text.split("ASSISTANT:")[-1].strip()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return response

def prepare_message(image_path, question, prev_qa=None):
    """准备LLaVA模型的消息格式"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 构建消息
    if prev_qa:
        # 多轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prev_qa["question"]},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": prev_qa["answer"]}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
    else:
        # 单轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            }
        ]
    
    return messages