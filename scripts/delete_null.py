import os
import json
import csv
from typing import Dict, List


def clean_json_file(file_path: str) -> None:
    """清理JSON文件中answer为空字符串的question"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "conversation" not in data:
            return

        original_len = len(data["conversation"])
        # 过滤空答案的问题
        filtered_conversation = []
        for qa in data["conversation"]:
            # 修改判断条件：仅当answer是空字符串时删除
            if qa.get("answer") == "":
                print(f"删除问题 - 文件: {file_path}")
                print(f"Turn ID: {qa.get('turn_id')}")
                print(f"Question: {qa.get('question')}")
                print("------------------------")
            else:
                filtered_conversation.append(qa)

        # 如果有问题被删除，更新文件
        if len(filtered_conversation) < original_len:
            data["conversation"] = filtered_conversation
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"文件已更新: {file_path}")
            print(f"删除了 {original_len - len(filtered_conversation)} 个问题")
            print("========================")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

def clean_csv_file(file_path: str) -> None:
    """删除CSV文件中model_answer为空的记录"""
    try:
        # 读取CSV文件
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            original_fieldnames = reader.fieldnames
            for row in reader:
                if row["ground_truth"].strip() != "":
                    rows.append(row)
                else:
                    print(f"删除记录 - 文件: {file_path}")
                    print(f"Turn ID: {row['turn_id']}")
                    print(f"Question: {row['question']}")
                    print("------------------------")

        # 如果有记录被删除
        if len(rows) < sum(1 for _ in open(file_path, encoding="utf-8")) - 1:
            # 保存更新后的CSV文件
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=original_fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"文件已更新: {file_path}")
            print("========================")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")


def process_directory(root_dir: str) -> None:
    """处理目录下的所有JSON文件"""
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json") and file != "cognitive_map.json":
                file_path = os.path.join(root, file)
                clean_json_file(file_path)


def main():
    # testdata_dir = os.path.join(os.path.dirname(__file__), "..", "testdata")
    # if not os.path.exists(testdata_dir):
    #     print(f"目录不存在: {testdata_dir}")
    #     return
    # print("开始处理文件...")
    # process_directory(testdata_dir)
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    csv_file_path = os.path.join(result_dir, "gemini-pro-1.5_all_results_vanilla_qa_1_fixed.csv")
    clean_csv_file(csv_file_path)

    print("处理完成")


if __name__ == "__main__":
    main()
