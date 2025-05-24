import json
import csv
import os
import argparse
from typing import Dict, List
from tqdm import tqdm
import sys
import torch
from PIL import Image
import socket
import logging
import debugpy
import importlib

# Import evaluation metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from evaluation.metric import evaluate_with_rotations, evaluation_re


def setup_remote_debugging(port: int = 2233):
    """设置远程调试"""
    ip = socket.gethostbyname(socket.gethostname())
    logging.info(f"初始化远程调试: {ip}:{port}")
    print(f"初始化远程调试: {ip}:{port}")

    debugpy.listen(address=(ip, port))
    debugpy.wait_for_client()
    logging.info(f"调试器已连接: {debugpy.is_client_connected()}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用多模态VLM模型评估自制数据集")
    parser.add_argument("--input_dir", type=str, required=True, help="输入JSON文件目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen", "llama", "deepseek", "janus", "internvl", "llava"],
        required=True,
        help="模型类型: qwen、llama、deepseek或janus",
    )
    parser.add_argument("--image_base_dir", type=str, default="", help="图片基础目录")
    parser.add_argument("--json-suffix", type=str, required=True, help="JSON文件后缀")
    parser.add_argument(
        "--device", type=str, default="cuda", help="运行设备，cuda或cpu"
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="A40",
        choices=["A40", "A800"],
        help="GPU类型，影响模型分割策略（DeepSeek-VL2需要）",
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="使用Flash Attention"
    )
    parser.add_argument("--debug", action="store_true", help="启用远程调试")
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="vanilla",
        choices=["vanilla", "think"],
        help="选择使用的提示模板类型：vanilla或think",
    )
    parser.add_argument(
        "--start_folder",
        type=str,
        default="",
        help="指定开始处理的文件夹名称，之前的文件夹将被跳过",
    )
    return parser.parse_args()


def load_json_data(file_path: str) -> Dict:
    """加载并解析JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_json_files_with_suffix(root_dir: str, suffix: str) -> List[str]:
    """查找指定后缀的JSON文件"""
    json_files = []

    if not os.path.exists(root_dir):
        print(f"目录不存在: {root_dir}")
        return []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(f"_{suffix}.json"):
                json_files.append(os.path.join(folder_path, file))

    return json_files


def save_results_to_csv(results: List[Dict], output_file: str):
    """保存结果到CSV文件"""
    if not results:
        print("没有结果可保存")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_results = []
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())

    fieldnames = list(all_fields)
    for result in results:
        processed_result = {}
        for field in fieldnames:
            value = result.get(field, "")
            if isinstance(value, (dict, list)):
                processed_result[field] = json.dumps(value, ensure_ascii=False)
            else:
                processed_result[field] = value
        processed_results.append(processed_result)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=processed_results[0].keys())
        writer.writeheader()
        writer.writerows(processed_results)

    print(f"结果已保存到 {output_file}")


def get_model_display_name(model_type, model_path):
    """获取模型显示名称"""
    if model_type == "qwen":
        return "Qwen2.5-VL-72B-Instruct"
    elif model_type == "llama":
        return "Llama-3.2-90B-Vision-Instruct"
    elif model_type == "deepseekvl":
        if "small" in model_path.lower():
            return "DeepSeek-VL2-Small"
        else:
            return "DeepSeek-VL2"
    elif model_type == "janus":
        return "Janus-Pro-7B"
    elif model_type == "internvl":
        return "InternVL2_5-78B"
    elif model_type == "llava":
        if "v1.5-13b" in model_path.lower():
            return "LLaVA-v1.5-13B"
        else:
            return "LLaVA"
    else:
        return model_path.split("/")[-1]


def find_image_path(image_id, json_path, image_base_dir):
    """查找有效的图片路径"""
    # 直接路径
    if os.path.exists(image_id):
        return image_id

    # JSON文件目录下的rgb.png
    if json_path:
        json_dir = os.path.dirname(json_path)
        alternative_path = os.path.join(json_dir, "rgb.png")
        if os.path.exists(alternative_path):
            print(f"使用替代图片路径: {alternative_path}")
            return alternative_path

    # 基础目录路径
    if image_base_dir:
        parts = image_id.split("/")
        if len(parts) >= 2:
            folder_name = parts[-2]
            file_name = parts[-1]
            alternative_path = os.path.join(image_base_dir, folder_name, file_name)
            if os.path.exists(alternative_path):
                print(f"使用基础目录图片路径: {alternative_path}")
                return alternative_path

    return image_id


def run_vqa_test(
    json_data: Dict,
    model_info,
    device="cuda",
    image_base_dir="",
    json_path="",
    model_type="qwen",
    prompt_type="vanilla",
):
    """执行VQA测试并返回结果"""
    # 根据模型类型动态导入相应模块
    generate_module = importlib.import_module(f"load_{model_type}_generate")

    results = []
    image_path = json_data.get("image_id", "")
    conversation = json_data.get("conversation", [])
    pre_prompt = "According to the PANORAMA and the predicted objects' center locations, answer the following question:"
    post_prompt_vanilla = """
    Please answer the question with a brief response following these guidelines:
    - For counting questions: Respond with only an integer (including 0 if none are present).
    - For closest object questions: Respond with just the name of the closest object, or the exact error message if applicable.
    - For directional questions: Respond with only the direction (front, back, left, right, front-left, front-right, back-left, or back-right), capitalized correctly, or the exact error message if applicable.
    Do not include explanations or additional text in your answer. 
    """
    post_prompt_think = """
    Please think step by step and enclose your complete reasoning process in <think> </think> tags. 
    Within your thinking process:
    - For counting questions: Identify all instances of the queried object(s) one by one.
    - For closest object questions: Identify the positioning object and all candidate objects, then compare their relative distances.
    - For directional questions: Locate the referenced objects and determine spatial relationships carefully.
    After your thinking, provide ONLY your final answer within <answer> </answer> tags using:
    - For counting questions: Just an integer (including 0 if none are present).
    - For closest object questions: Only the name of the closest object, or the exact error message if applicable.
    - For directional questions: Only the direction (Front, Back, Left, Right, Front-Left, Front-Right, Back-Left, or Back-Right) with correct capitalization, or the exact error message if applicable.
    """

    # 根据prompt_type选择提示模板
    selected_prompt = (
        post_prompt_think if prompt_type == "think" else post_prompt_vanilla
    )

    # 查找有效图片路径
    full_image_path = find_image_path(image_path, json_path, image_base_dir)

    # 检查图片是否存在
    if not os.path.exists(full_image_path):
        print(f"警告: 无法找到图片，尝试了路径: {full_image_path}")
        return []

    # 加载图像
    try:
        image = Image.open(full_image_path).convert("RGB")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return []

    # 获取模型和处理器
    model = model_info.get("model")
    processor = model_info.get("processor")
    tokenizer = model_info.get("tokenizer")
    model_name = get_model_display_name(model_type, model_info.get("model_path", ""))

    # 获取模型特定的消息准备和生成函数
    prepare_message = generate_module.prepare_message
    
    if model_type in ["deepseekvl", "janus", "internvl"]:
        generate_func = lambda m, p, t, msg: generate_module.generate(m, p, t, msg, device)
    else:
        generate_func = lambda m, p, msg: generate_module.generate(m, p, msg, device)

    # 获取第一轮问题
    first_turn = next((q for q in conversation if q.get("turn_id") == 0), None)
    first_turn_response = None
    first_turn_score = None

    # 如果有第一轮问题，先处理
    if first_turn:
        first_question = first_turn.get("question", "")
        first_ground_truth = first_turn.get("answer", "")
        print(f"处理第一轮问题...")

        # 准备模型消息
        first_messages = prepare_message(full_image_path, first_question) #image
        
        # 运行模型获取第一轮回答
        try:
            if model_type in ["deepseekvl", "janus", "internvl"]:
                first_turn_response = generate_func(model, processor, tokenizer, first_messages)
            else:
                first_turn_response = generate_func(model, processor, first_messages)

            # 计算第一轮得分
            first_turn_score, rotation = evaluate_with_rotations(
                first_ground_truth, first_turn_response
            )

            # 添加第一轮结果
            results.append(
                {
                    "image_id": image_path,
                    "model": model_name,
                    "turn_id": 0,
                    "skills_tested": first_turn.get("skills_tested", "cognitive_map"),
                    "question": first_question,
                    "model_answer": first_turn_response,
                    "ground_truth": str(first_ground_truth),
                    "score": first_turn_score,
                }
            )
        except Exception as e:
            print(f"处理第一轮问题时出错: {e}")
            results.append({
                "image_id": image_path,
                "model": model_name,
                "turn_id": 0,
                "skills_tested": first_turn.get("skills_tested", "cognitive_map"),
                "question": first_question,
                "model_answer": first_turn_response,
                "ground_truth": str(first_ground_truth),
                "score": first_turn_score
            })
    
    # 处理每个第二轮问题
    for turn in tqdm(conversation):
        if turn.get("turn_id") == 0:
            continue  # 跳过第一轮问题

        turn_id = turn.get("turn_id")
        question = turn.get("question", "")
        ground_truth = turn.get("answer", "")
        skill_type = turn.get("skills_tested", turn.get("follow_up", "unknown"))

        print(f"\n处理问题 {turn_id}: {question[:50]}...")

        try:
            # 准备问题文本
            query = f"{question} {selected_prompt}"

            # 根据是否有第一轮对话准备消息
            if first_turn_response:
                # 多轮对话
                prev_qa = {
                    "question": first_turn.get("question", ""),
                    "answer": first_turn_response,
                }
                messages = prepare_message(full_image_path, query, prev_qa)
            else:
                # 单轮对话
                full_query = f"{pre_prompt} {query}"
                messages = prepare_message(full_image_path, full_query)

            # 运行模型获取回答
            if model_type in ["deepseekvl", "janus", "internvl"]:
                model_answer = generate_func(model, processor, tokenizer, messages)
            else:
                model_answer = generate_func(model, processor, messages)

            # 计算得分
            second_turn_score = evaluation_re(
                ground_truth, model_answer, skill=skill_type, use_think_prompt=(prompt_type == "think")
            )

        except Exception as e:
            print(f"处理问题 {turn_id} 时出错: {e}")
            model_answer = f"ERROR: {str(e)}"
            second_turn_score = 0.0

        # 记录结果
        result = {
            "image_id": image_path,
            "model": model_name,
            "turn_id": turn_id,
            "skills_tested": skill_type,
            "question": question,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "score": second_turn_score,
        }

        results.append(result)

    return results


def process_file(
    file_path,
    model_info,
    device,
    output_dir,
    image_base_dir="",
    json_suffix="",
    model_type="qwen",
    prompt_type="vanilla",
):
    """处理单个JSON文件的测试"""
    try:
        print(f"处理文件: {file_path}")
        json_data = load_json_data(file_path)
        file_name = os.path.basename(file_path).split(".")[0]

        model_name = get_model_display_name(
            model_type, model_info.get("model_path", "")
        )
        print(f"\n开始使用{model_name}测试...")

        # 执行测试
        results = run_vqa_test(
            json_data,
            model_info,
            device,
            image_base_dir,
            file_path,
            model_type,
            prompt_type,
        )

        if results:
            # 保存单个文件的结果
            model_dir = os.path.join(
                output_dir, f"{model_name}_{json_suffix}_{prompt_type}"
            )
            os.makedirs(model_dir, exist_ok=True)
            output_file = os.path.join(model_dir, f"{file_name}_results.csv")

            save_results_to_csv(results, output_file)
            
            # # 更新汇总文件
            # summary_file = os.path.join(output_dir, f"{model_name}_all_results_{prompt_type}_{json_suffix}.csv")
            
            # if os.path.exists(summary_file):
            #     # 读取现有结果
            #     existing_results = []
            #     with open(summary_file, 'r', encoding='utf-8') as f:
            #         reader = csv.DictReader(f)
            #         existing_results = list(reader)
                
            #     # 合并结果
            #     all_results = existing_results + results
            #     save_results_to_csv(all_results, summary_file)
            # else:
            #     # 创建新的汇总文件
            #     save_results_to_csv(results, summary_file)
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def load_model_by_type(
    model_type, model_path, device, gpu_type="A40", use_flash_attention=True
):
    """根据模型类型加载模型和处理器"""
    # 动态导入相应的模型加载模块
    try:
        module_name = f"evaluation.model_run.load_{model_type}_generate"
        model_module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"无法导入模型模块 {module_name}: {e}")

    model_info = {"model_path": model_path}

    if model_type == "qwen":
        model, processor = model_module.load_model(
            model_path, device, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor

    elif model_type == "llama":
        model, processor = model_module.load_model(
            model_path, device, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor
    
    elif model_type == "deepseekvl":
        model, processor, tokenizer = model_module.load_model(
            model_path, device, gpu_type, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor
        model_info["tokenizer"] = tokenizer

    elif model_type == "janus":
        model, processor, tokenizer = model_module.load_model(
            model_path, device, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor
        model_info["tokenizer"] = tokenizer

    elif model_type == "internvl":
        model, processor, tokenizer = model_module.load_model(
            model_path, device, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor
        model_info["tokenizer"] = tokenizer

    elif model_type == "llava":
        model, processor, tokenizer = model_module.load_model(
            model_path, device, use_flash_attention
        )
        model_info["model"] = model
        model_info["processor"] = processor

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model_info


def main():
    args = parse_args()

    if args.debug:
        setup_remote_debugging()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型（只加载指定类型的模型）
    model_info = load_model_by_type(
        args.model_type,
        args.model_path,
        args.device,
        args.gpu_type,
        args.use_flash_attention,
    )

    # 查找符合条件的JSON文件
    json_files = find_json_files_with_suffix(args.input_dir, args.json_suffix)

    if not json_files:
        print(f"未找到后缀为 {args.json_suffix} 的JSON文件")
        return
    
    if args.start_folder:
        filtered_files = []
        start_processing = False
        # 按文件夹名称排序
        folders = sorted(os.listdir(args.input_dir))

        for folder in folders:
            if folder == args.start_folder:
                start_processing = True

            if start_processing:
                folder_path = os.path.join(args.input_dir, folder)
                for json_file in json_files:
                    if json_file.startswith(folder_path):
                        filtered_files.append(json_file)

        json_files = filtered_files
        print(
            f"从文件夹 {args.start_folder} 开始处理，共找到 {len(json_files)} 个符合条件的JSON文件"
        )
    else:
        print(f"找到 {len(json_files)} 个符合条件的JSON文件")
    
    # 处理所有JSON文件
    for file_path in json_files:
        process_file(
            file_path,
            model_info,
            args.device,
            args.output_dir,
            args.image_base_dir,
            args.json_suffix,
            args.model_type,
            args.prompt_type,
        )

    print("所有测试完成")


if __name__ == "__main__":
    main()
