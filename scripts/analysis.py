import pandas as pd
import numpy as np
import json
import glob


def analyze_scores(csv_path):
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 计算整体得分
    overall_scores = []
    for skill in df["skills_tested"].unique():
        skill_data = df[df["skills_tested"] == skill]

        if skill == "cognitive_map":
            # 提取cognitive_map的F1得分
            f1_scores = []
            for index, row in skill_data.iterrows():
                try:
                    score_json = row["score"]
                    # 检查是否为字符串类型
                    if not isinstance(score_json, str):
                        print(
                            f"警告: 行 {index+2} (skill={skill})中的score不是字符串，而是 {type(score_json)}: {score_json}"
                        )
                        continue

                    score_dict = json.loads(score_json)
                    f1_score = score_dict.get("overall", {}).get("f1", None)
                    if f1_score is not None:
                        f1_scores.append(f1_score)
                except Exception as e:
                    print(f"错误: 无法解析行 {index+2} (skill={skill})中的JSON: {e}")
                    print(f"问题数据: {score_json}")
                    continue

            if f1_scores:
                skill_avg = np.mean(f1_scores)
                overall_scores.append(skill_avg)
        else:
            # 其他技能类型的得分
            scores = pd.to_numeric(skill_data["score"], errors="coerce")
            skill_avg = scores.mean()
            overall_scores.append(skill_avg)

    # 计算所有技能的平均得分作为整体得分
    overall_score = np.mean(overall_scores) if overall_scores else 0
    print(f"\n整体得分: {overall_score:.4f}")

    # 按skills_tested分组进行分析
    for skill in df["skills_tested"].unique():
        skill_data = df[df["skills_tested"] == skill]

        if skill == "cognitive_map":
            # 处理cogmap类型的得分，需要从JSON中提取数据
            metrics = []
            for index, row in skill_data.iterrows():
                try:
                    score_json = row["score"]
                    # 检查是否为字符串类型
                    if not isinstance(score_json, str):
                        print(
                            f"跳过行 {index+2} (skill={skill})中的非字符串score: {type(score_json)}"
                        )
                        continue

                    score_dict = json.loads(score_json)
                    metrics.append(
                        {
                            "precision": score_dict.get("overall", {}).get(
                                "precision", None
                            ),
                            "recall": score_dict.get("overall", {}).get("recall", None),
                            "f1": score_dict.get("overall", {}).get("f1", None),
                            "avg_distance": score_dict.get("overall", {}).get(
                                "avg_distance", None
                            ),
                            "tp": score_dict.get("overall", {}).get("tp", None),
                            "fp": score_dict.get("overall", {}).get("fp", None),
                            "fn": score_dict.get("overall", {}).get("fn", None),
                            "chair_s": score_dict.get("overall", {}).get(
                                "chair_s", None
                            ),
                            "chair_i": score_dict.get("overall", {}).get(
                                "chair_i", None
                            ),
                            "chair_instance": score_dict.get("overall", {}).get(
                                "chair_instance", None
                            ),
                        }
                    )
                except Exception as e:
                    print(f"错误: 无法解析行 {index+2} (skill={skill})中的JSON: {e}")
                    print(f"问题数据: {score_json}")
                    continue

            # 将提取的指标转换为DataFrame
            if metrics:
                metrics_df = pd.DataFrame(metrics)

                # 筛选出成功生成cogmap的记录(通过检查per_class不为空)
                valid_cogmaps = []
                for index, row in skill_data.iterrows():
                    try:
                        score_dict = json.loads(row["score"])
                        if score_dict.get("per_class"):  # 检查per_class是否为非空
                            valid_cogmaps.append(True)
                        else:
                            valid_cogmaps.append(False)
                    except:
                        valid_cogmaps.append(False)

                metrics_df["valid_cogmap"] = valid_cogmaps

                print(f"\n分析结果 - {skill}:")
                print("统计量概要：")
                print(metrics_df.describe())

                # 分别计算所有记录和有效cogmap记录的指标
                for column in metrics_df.columns:
                    if column == "valid_cogmap":
                        continue

                    # 所有记录的统计
                    all_mean = metrics_df[column].mean()
                    all_std = metrics_df[column].std()

                    # 仅有效cogmap记录的统计
                    valid_data = metrics_df[metrics_df["valid_cogmap"]]
                    valid_mean = (
                        valid_data[column].mean()
                        if not valid_data.empty
                        else float("nan")
                    )
                    valid_std = (
                        valid_data[column].std()
                        if not valid_data.empty
                        else float("nan")
                    )

                    print(f"\n{column}:")
                    print(f"  所有记录 - 均值: {all_mean:.4f}, 标准差: {all_std:.4f}")
                    print(
                        f"  仅有效cogmap - 均值: {valid_mean:.4f}, 标准差: {valid_std:.4f}"
                    )

                # 输出有效/无效cogmap的比例
                total = len(metrics_df)
                valid = metrics_df["valid_cogmap"].sum()
                print(f"\nCogmap生成统计:")
                print(f"  总记录数: {total}")
                print(f"  有效cogmap数: {valid}")
                print(f"  生成成功率: {(valid/total*100):.2f}%")
            else:
                print(f"\n分析结果 - {skill}: 没有有效的指标数据")

        else:
            # 处理其他类型的得分
            print(f"\n分析结果 - {skill}:")
            print("得分统计量：")
            scores = pd.to_numeric(skill_data["score"], errors="coerce")
            print(scores.describe())

    # 计算除cognitive_map外的整体统计
    non_cogmap_data = df[df["skills_tested"] != "cognitive_map"]
    non_cogmap_scores = pd.to_numeric(non_cogmap_data["score"], errors="coerce")
    print("\n除cognitive_map外所有技能的整体统计：")
    print(non_cogmap_scores.describe())
    print(f"整体均分: {non_cogmap_scores.mean():.4f}")


def combine_csv_files(csv_folder="result/gemini-pro-1.5"):
    # 获取所有CSV文件
    csv_files = glob.glob(f"{csv_folder}/*.csv")

    if not csv_files:
        print("在目录下未找到CSV文件")
        return None

    print(f"找到{len(csv_files)}个CSV文件")

    # 合并所有CSV文件
    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"已加载: {file}")
        except Exception as e:
            print(f"读取{file}出错: {e}")

    if not all_dfs:
        print("没有有效的CSV文件可以合并")
        return None

    # 合并所有DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 保存合并后的CSV
    output_path = f"{csv_folder}_all.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"已将合并结果保存到: {output_path}")

    return output_path


# 使用示例
if __name__ == "__main__":
    # combine_csv_files("result/qwen2.5-vl-72b-instruct_qa_with_negative_vanilla")
    csv_path = "result/internvl3-14b_free_qa_vanilla_all.csv"  # 替换为你的CSV文件路径
    analyze_scores(csv_path)
