import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def split_model(model_name, gpu_type="A40"):
    """
    根据GPU类型分割DeepSeek模型
    
    Args:
        model_name: 模型路径
        gpu_type: GPU类型 ("A40"或"A800")
    
    Returns:
        模型分布的设备映射
    """
    device_map = {}
    
    if gpu_type == "A800":
        # A800用2个GPU
        if "deepseek-vl2-small" in model_name:  # 小模型(16B)
            num_layers_per_gpu = [13, 13]  # 在2个GPU上平均分配层
        else:  # 完整模型(27B)
            num_layers_per_gpu = [15, 15]  # 在2个GPU上平均分配层
    else:
        # A40用3个GPU来处理完整模型
        if "deepseek-vl2-small" in model_name:  # 小模型(16B)
            num_layers_per_gpu = [13, 14]  # 2个GPU处理16B
        else:  # 完整模型(27B)
            num_layers_per_gpu = [10, 10, 10]  # 3个GPU处理27B
    
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    
    # 所有这些组件放在第一个GPU (设备0)上
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    
    return device_map

def load_model(model_path, device='cuda', gpu_type="A40", use_flash_attention=True):
    """
    加载DeepSeek-VL2模型和处理器
    """
    print(f"正在加载DeepSeek-VL2模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 获取模型分割的设备映射
    device_map = split_model(model_path, gpu_type)
    
    # 加载处理器
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    # 加载模型
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    
    print(f"DeepSeek-VL2模型加载完成")
    return model, processor, tokenizer

def generate(model, processor, tokenizer, messages, device='cuda'):
    """
    使用DeepSeek-VL2模型生成回答
    """
    # 加载图像并准备输入
    pil_images = load_pil_images(messages)
    
    # 添加系统提示（如果需要）
    prepare_inputs = processor(
        conversations=messages,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device)
    
    # 运行图像编码器获取图像嵌入
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    
    # 运行模型生成回答
    with torch.no_grad():
        outputs = model.language.generate(
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
    """准备DeepSeek-VL2模型的消息格式"""
    if prev_qa:
        # 多轮对话
        messages = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prev_qa['question']}",
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
                "content": f"<image>\n<|grounding|>{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    return messages