import os
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载Llama-3.2-90B-Vision-Instruct模型和处理器"""
    print(f"正在加载模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 基于use_flash_attention参数决定如何加载模型
    if not use_flash_attention:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
    else:
        # 使用Flash Attention 2进行内存优化
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  
            attn_implementation="flash_attention_2",  
            device_map="auto"  
        )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"模型加载完成")
    return model, processor

def generate(model, processor, messages, device='cuda'):
    """使用Llama-3.2-90B-Vision模型生成回答"""
    # 准备输入
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    
    # 处理图像和文本内容 - 只从第一个用户消息中提取图像
    image_input = None
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            # 检查content是否为列表
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_input = item["image"]
                        break
            if image_input:
                break
    
    if image_input is None:
        raise ValueError("未找到图像输入")
    
    # 使用混合精度进行推理以减少内存使用
    with torch.cuda.amp.autocast():
        inputs = processor(
            image_input,
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        
        # 使用内存高效的设置进行生成
        with torch.no_grad():  
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512
            )
        
        # 处理输出
        output_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return output_text

def prepare_message(image_path, question, prev_qa=None):

    image = Image.open(image_path)

    """准备Llama模型的消息格式"""
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