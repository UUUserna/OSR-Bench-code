import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """构建图像变换"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近的宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态预处理图像"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    """加载并预处理图像（InternVL格式）"""
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        # 如果是其他格式，尝试转换
        image = Image.fromarray(image_file).convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model(model_path, device='cuda', use_flash_attention=True):
    """加载 InternVL 模型和处理器"""
    print(f"正在加载 InternVL 模型: {model_path}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    print(f"InternVL 模型加载完成")
    # 返回模型、None（处理器）、分词器，保持与其他模型一致的接口
    return model, None, tokenizer

def generate(model, processor, tokenizer, messages, device='cuda'):
    """使用 InternVL 模型生成回答"""
    try:
        # 从消息中提取文本和图像
        text = ""
        image = None
        pixel_values = None
        history = None
        
        # 处理消息格式
        if len(messages) == 1:
            # 单轮对话
            msg = messages[0]
            for content in msg["content"]:
                if content["type"] == "image":
                    image = content["source"]
                elif content["type"] == "text":
                    text = content["text"]
            
            # 处理图像
            if image is not None:
                pixel_values = load_image_internvl(image).to(torch.bfloat16).to(device)
                # 在问题前添加图像标记
                text = f"<image>\n{text}"
                
        else:
            # 多轮对话
            # 从第一个消息中提取图像
            first_msg = messages[0]
            for content in first_msg["content"]:
                if content["type"] == "image":
                    image = content["source"]
                    break
            
            # 处理图像
            if image is not None:
                pixel_values = load_image_internvl(image).to(torch.bfloat16).to(device)
            
            # 构建历史记录
            history = []
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    q = ""
                    a = ""
                    # 提取用户问题
                    user_msg = messages[i]
                    for content in user_msg["content"]:
                        if content["type"] == "text":
                            q = content["text"]
                    
                    # 第一轮需要添加图像标记
                    if i == 0 and image is not None:
                        q = f"<image>\n{q}"
                    
                    # 提取助手回答
                    assistant_msg = messages[i+1]
                    if isinstance(assistant_msg["content"], list):
                        for content in assistant_msg["content"]:
                            if content["type"] == "text":
                                a = content["text"]
                    else:
                        a = assistant_msg["content"]
                    
                    if q and a:
                        history.append((q, a))
            
            # 处理最后一个用户消息
            last_msg = messages[-1]
            for content in last_msg["content"]:
                if content["type"] == "text":
                    text = content["text"]
        
        # 确保text不为空
        if not text or text.strip() == "":
            text = "Describe this picture."
        
        # 生成配置
        generation_config = dict(max_new_tokens=512, do_sample=False)
        
        # 使用chat方法生成回答
        response = model.chat(
            tokenizer, 
            pixel_values, 
            text, 
            generation_config, 
            history=history,
            return_history=False
        )
        
        return response
    
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