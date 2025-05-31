import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载LLaVA模型和处理器"""
    print(f"正在加载LLaVA模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 获取模型名称
    model_name = get_model_name_from_path(model_path)
    
    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device
    )
    
    print(f"LLaVA模型加载完成")
    # 返回模型、处理器（image_processor）和tokenizer，保持接口一致
    return model, image_processor, tokenizer

def generate(model, processor, tokenizer, messages, device='cuda'):
    """使用LLaVA模型生成回答"""
    try:
        # 提取图像和文本
        image = None
        conversation_text = []
        
        # 获取模型名称以确定对话模式
        if hasattr(model.config, '_name_or_path'):
            model_name = model.config._name_or_path
        else:
            model_name = "llava-v1.5"
        
        # 根据模型版本选择对话模板
        if "v1.6" in model_name or "34b" in model_name:
            conv_mode = "chatml_direct"
        elif "v1.5" in model_name:
            conv_mode = "llava_v1"
        elif "v1" in model_name:
            conv_mode = "llava_v1"
        else:
            conv_mode = "llava_v0"
        
        # 创建对话对象
        conv = conv_templates[conv_mode].copy()
        
        # 处理消息
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                user_content = ""
                for content in msg["content"]:
                    if content["type"] == "text":
                        user_content += content["text"]
                    elif content["type"] == "image":
                        image = content["source"]
                        # 在第一个用户消息中添加图像标记
                        if i == 0:
                            if model.config.mm_use_im_start_end:
                                user_content = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_content
                            else:
                                user_content = DEFAULT_IMAGE_TOKEN + '\n' + user_content
                conv.append_message(conv.roles[0], user_content)
            elif msg["role"] == "assistant":
                assistant_content = ""
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content["type"] == "text":
                            assistant_content += content["text"]
                else:
                    assistant_content = msg["content"]
                conv.append_message(conv.roles[1], assistant_content)
        
        # 添加一个空的assistant消息以触发生成
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # 使用image_processor处理图像
            image_tensor = process_images([image], processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(device, dtype=torch.float16)
        else:
            image_tensor = None
        
        # Tokenize输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        
        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True
            )
        
        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 提取生成的回答部分
        # 移除输入部分，只保留生成的内容
        if conv.sep_style == SeparatorStyle.TWO:
            # 对于使用两个分隔符的对话模式
            sep2 = conv.sep2
            response = outputs.split(sep2)[-1].strip()
        else:
            # 尝试通过角色名分割
            response = outputs.split(conv.roles[1] + ":")[-1].strip()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return response
    
    except Exception as e:
        print(f"生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return f"生成回答时出错: {str(e)}"

def prepare_message(image_path, question, prev_qa=None):
    """准备LLaVA模型的消息格式"""
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"加载图像失败: {e}")
        raise
    
    # 确保问题不为空
    if not question or question.strip() == "":
        question = "What do you see in this image?"
    
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