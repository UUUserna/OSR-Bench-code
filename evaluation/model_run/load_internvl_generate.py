import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载 InternVL 模型和处理器"""
    print(f"正在加载 InternVL 模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 加载处理器和分词器
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    print(f"InternVL 模型加载完成")
    return model, processor, tokenizer

def generate(model, processor, tokenizer, messages, device='cuda'):
    """使用 InternVL 模型生成回答"""
    try:
        # 从消息中提取文本和图像
        text = ""
        image = None
        
        for msg in messages:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content["type"] == "image":
                        image = content["source"]
                    elif content["type"] == "text":
                        text += content["text"] + " "
        
        # 确保text不为空
        if not text or text.strip() == "":
            text = "请描述这张图片。"
        
        # 使用processor处理输入 - 明确提供text参数
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        
        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 解码回答
        answer = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return answer
    
    except Exception as e:
        print(f"生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return f"生成回答时出错: {str(e)}"

def prepare_message(image_path, question, prev_qa=None):
    """准备 InternVL 模型的消息格式"""
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"加载图像失败: {e}")
        raise
    
    # 确保问题不为空
    if not question or question.strip() == "":
        question = "describe this picture."
    
    # 构建消息
    if prev_qa:
        # 多轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "text", "text": prev_qa["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": prev_qa["answer"]}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
    else:
        # 单轮对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
    
    return messages