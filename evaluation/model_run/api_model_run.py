import json
import csv
import os
import requests
import base64
import time
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from evaluation.metric import evaluate_with_rotations, evaluation_re
from evaluation.llm_evaluator import evaluate_with_openai

# ${workspaceFolder}/evaluation/model_run/api_model_run.py

# OpenRouter API配置
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = ""  # 替换为你的实际API密钥
DEEPSEEK_API_KEY = ""


def parse_args():
    parser = argparse.ArgumentParser(description="使用OpenRouter API测试VQA任务")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="输入JSON文件目录(data或testdata)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="要测试的模型，如 openai/gpt-4o-2024-11-20、anthropic/claude-3.7-sonnet、google/gemini-pro-1.5",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="",
        help="图片基础目录，如果不提供则使用JSON中的路径",
    )
    parser.add_argument(
        "--json-suffix",
        type=str,
        required=True,
        help="指定JSON文件后缀，如qa, qa_with_negative, qa_without_cogmap或qa_with_negative_without_cogmap",
    )
    parser.add_argument(
        "--start_folder",
        type=str,
        default="",
        help="指定开始处理的文件夹名称，之前的文件夹将被跳过",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="vanilla",
        choices=["vanilla", "think"],
        help="选择使用的提示模板类型：vanilla或think",
    )
    return parser.parse_args()


def load_json_data(file_path: str) -> Dict:
    """加载并解析JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image(image_path: str) -> str:
    """将图片编码为base64格式"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def call_openrouter_api(model: str, messages: List[Dict], max_retries: int = 6) -> Dict:
    """调用OpenRouter API获取模型响应"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # "HTTP-Referer": "http://localhost",  # 可选，用于OpenRouter排名
        # "X-Title": "VQA-Testing"  # 可选，用于OpenRouter排名
    }

    payload = {"model": model, "messages": messages, "temperature": 0.0}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL, headers=headers, data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 指数退避策略
                print(f"API调用失败，{wait_time}秒后重试: {e}")
                time.sleep(wait_time)
            else:
                print(f"API调用达到最大重试次数: {e}")
                return {"error": str(e)}

    return {"error": "达到最大重试次数"}


def run_vqa_test(
    json_data: Dict,
    model: str,
    image_base_dir: str = "",
    json_path: str = "",
    suffix: str = "",
    prompt_type: str = "vanilla",
) -> List[Dict]:
    """执行VQA测试并返回结果"""
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

    # 处理图片路径
    # 优先使用json_data中的image_id
    full_image_path = image_path

    # 如果image_id路径不存在，尝试使用json文件所在目录下的rgb.png
    if not os.path.exists(full_image_path) and json_path:
        json_dir = os.path.dirname(json_path)
        alternative_path = os.path.join(json_dir, "rgb.png")
        if os.path.exists(alternative_path):
            full_image_path = alternative_path
            print(f"使用替代图片路径: {full_image_path}")

    # 如果提供了image_base_dir，再尝试在基础目录中查找
    if not os.path.exists(full_image_path) and image_base_dir:
        # 从image_id中提取文件夹名称和文件名
        parts = image_path.split("/")
        if len(parts) >= 2:
            folder_name = parts[-2]
            file_name = parts[-1]
            alternative_path = os.path.join(image_base_dir, folder_name, file_name)
            if os.path.exists(alternative_path):
                full_image_path = alternative_path
                print(f"使用基础目录图片路径: {full_image_path}")

    # 最终检查图片是否存在
    if not os.path.exists(full_image_path):
        print(f"警告: 无法找到图片，尝试了路径: {image_path}")
        return []

    # 编码图片
    try:
        image_base64 = encode_image(full_image_path)
    except Exception as e:
        print(f"编码图片失败: {e}")
        return []

    # 获取第一轮问题(turn_id: 0)
    first_turn = next((q for q in conversation if q.get("turn_id") == 0), None)
    first_turn_response = None
    first_turn_score = None

    # 如果有第一轮问题，先处理它
    if first_turn:
        first_question = first_turn.get("question", "")
        first_ground_truth = first_turn.get("answer", "")
        print(f"处理第一轮问题...")

        # 构建第一轮消息
        first_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # 调用API获取第一轮回答
        first_response = call_openrouter_api(model, first_messages)

        if "error" in first_response:
            print(f"第一轮API调用失败: {first_response['error']}")
            # 记录第一轮失败信息
            with open("api_failures.log", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"API调用失败: 文件={json_path}, 问题ID=0, 问题={first_question}, 错误={first_response['error']}\n"
                )
            return []

        first_turn_response = first_response["choices"][0]["message"]["content"]

        # 使用hungarian_algorithm计算第一轮得分
        first_turn_score, rotation = evaluate_with_rotations(
            first_ground_truth, first_turn_response
        )

        # 添加第一轮结果
        results.append(
            {
                "image_id": image_path,
                "model": model,
                "turn_id": 0,
                "skills_tested": first_turn.get("skills_tested", "cognitive_map"),
                "question": first_question,
                "model_answer": first_turn_response,
                "ground_truth": str(first_ground_truth),
                "score": first_turn_score,
            }
        )

    # 处理每个第二轮问题
    for turn in tqdm(conversation):
        if turn.get("turn_id") == 0:
            continue  # 跳过第一轮问题

        turn_id = turn.get("turn_id")
        question = turn.get("question", "")
        ground_truth = turn.get("answer", "")
        skill_type = turn.get("skills_tested", turn.get("follow_up", "unknown"))

        print(f"\n处理问题 {turn_id}: {question[:50]}...")

        # 根据prompt_type选择提示模板
        selected_prompt = (
            post_prompt_think if prompt_type == "think" else post_prompt_vanilla
        )

        # 构建消息列表
        messages = []

        if first_turn and first_turn_response:
            # 包含第一轮对话历史
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": first_turn.get("question", "")},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
                {"role": "assistant", "content": first_turn_response},
                {"role": "user", "content": question + " " + selected_prompt},
            ]
        else:
            # 直接问第二轮问题
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pre_prompt + question + " " + selected_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

        # 调用API获取回答
        response = call_openrouter_api(model, messages)

        # 检查API响应
        if "error" in response:
            model_answer = f"ERROR: {response['error']}"
            second_turn_score = 0.0

            # 记录失败信息
            with open("api_failures.log", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"API调用失败: 文件={json_path}, 问题ID={turn_id}, 问题={question}, 错误={response['error']}\n"
                )
        else:
            model_answer = response["choices"][0]["message"]["content"]
            # 使用evaluation函数计算第二轮得分
            if "with_negative" in suffix:
                second_turn_score_llm = evaluate_with_openai(
                    question,
                    ground_truth,
                    model_answer,
                    skill=skill_type,
                    api_key=DEEPSEEK_API_KEY,
                )
                second_turn_score = second_turn_score_llm.get("score", 0.0)
            else:
                second_turn_score = evaluation_re(
                    ground_truth,
                    model_answer,
                    skill=skill_type,
                    use_think_prompt=(prompt_type == "think"),
                )

            # 记录结果
            result = {
                "image_id": image_path,
                "model": model,
                "turn_id": turn_id,
                "skills_tested": skill_type,
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "score": second_turn_score,
            }

            # # 如果有第一轮得分，添加到结果中
            # if first_turn_score is not None:
            #     result["first_turn_score"] = first_turn_score

            results.append(result)

        # 避免API限流
        # time.sleep(1)

    return results


def save_results_to_csv(results: List[Dict], output_file: str):
    """保存结果到CSV文件"""
    if not results:
        print("没有结果可保存")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 处理嵌套数据
    processed_results = []
    for result in results:
        processed_result = {}
        for key, value in result.items():
            # 如果是复杂数据类型，转换为JSON字符串
            if isinstance(value, (dict, list)):
                processed_result[key] = json.dumps(value, ensure_ascii=False)
            else:
                processed_result[key] = value
        processed_results.append(processed_result)

    # 添加BOM以支持Excel正确显示中文
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=processed_results[0].keys())
        writer.writeheader()
        writer.writerows(processed_results)

    print(f"结果已保存到 {output_file}")


def find_json_files_with_suffix(root_dir: str, suffix: str) -> List[str]:
    """查找指定后缀的JSON文件"""
    json_files = []

    # 确保目录存在
    if not os.path.exists(root_dir):
        print(f"目录不存在: {root_dir}")
        return []

    # 遍历子目录
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 查找符合条件的JSON文件
        for file in os.listdir(folder_path):
            if file.endswith(f"_{suffix}.json"):
                json_files.append(os.path.join(folder_path, file))

    return json_files


def process_file(
    file_path: str,
    model: str,
    output_dir: str,
    image_base_dir: str = "",
    json_suffix: str = "",
    prompt_type: str = "vanilla",
):
    """处理单个JSON文件的测试"""
    try:
        print(f"处理文件: {file_path}")
        json_data = load_json_data(file_path)
        file_name = os.path.basename(file_path).split(".")[0]

        print(f"\n开始使用模型 {model} 测试...")
        # Replace colon with underscore in model name for file naming
        model_name = model.split("/")[-1].replace(":", "_")

        # 执行测试，传入prompt_type
        results = run_vqa_test(
            json_data, model, image_base_dir, file_path, json_suffix, prompt_type
        )

        if results:
            # 保存单个文件的结果
            model_dir = os.path.join(
                output_dir, f"{model_name}_{json_suffix}_{prompt_type}"
            )
            os.makedirs(model_dir, exist_ok=True)
            output_file = os.path.join(model_dir, f"{file_name}_results.csv")
            save_results_to_csv(results, output_file)

            # 更新汇总文件
            summary_file = os.path.join(
                output_dir, f"{model_name}_all_results_{prompt_type}_{json_suffix}.csv"
            )

            if os.path.exists(summary_file):
                # 读取现有结果
                existing_results = []
                with open(summary_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_results = list(reader)

                # 合并结果
                all_results = existing_results + results
                save_results_to_csv(all_results, summary_file)
            else:
                # 创建新的汇总文件
                save_results_to_csv(results, summary_file)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 查找符合条件的JSON文件
    json_files = find_json_files_with_suffix(args.input_dir, args.json_suffix)

    if not json_files:
        print(f"未找到后缀为 {args.json_suffix} 的JSON文件")
        return

    # 根据start_folder过滤文件
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
            args.model,
            args.output_dir,
            args.image_base_dir,
            args.json_suffix,
            args.prompt_type,
        )

    print("所有测试完成")


if __name__ == "__main__":
    main()
