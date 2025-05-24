import os
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def load_model(model_path, device='cuda', use_flash_attention=True):
    """
    加载Janus-Pro-7B模型和处理器
    """
    print(f"正在加载Janus-Pro模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 加载处理器和分词器
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    # 加载模型
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    
    print(f"Janus-Pro模型加载完成")
    return model, processor, tokenizer

def generate(model, processor, tokenizer, messages, device='cuda'):
    """
    使用Janus-Pro模型生成回答
    """
    # 加载图像并准备输入
    pil_images = load_pil_images(messages)
    
    prepare_inputs = processor(
        conversations=messages, 
        images=pil_images, 
        force_batchify=True
    ).to(device)
    
    # 运行图像编码器获取图像嵌入
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    # 运行模型生成回答
    with torch.no_grad():
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
    
    # 解码并返回回答
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # 清理内存
    torch.cuda.empty_cache()
    
    return answer

def prepare_message(image_path, question, prev_qa=None):
    """准备Janus-Pro模型的消息格式"""
    if prev_qa:
        # 多轮对话
        messages = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prev_qa['question']}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": prev_qa["answer"]},
            {"role": "<|User|>", "content": question},
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        # 单轮对话
        messages = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    return messages