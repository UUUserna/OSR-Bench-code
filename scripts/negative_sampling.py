import os
import json

def negative_sampling(folder_path, n, m):
    # 提取子文件夹名称前缀，例如 "ReplicaPano-large_apartment_0_006-Scene_Info-00057" 提取 "ReplicaPano"
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    prefix = folder_name.split('-')[0]
    
    # 组装文件路径
    cog_map_path = os.path.join(folder_path, 'cognitive_map.json')
    # neg_file_path = os.path.join(r"e:\00\PanoSpace\data", f"negative_sampling_{prefix}.json")
    neg_file_path = os.path.join(r"e:\00\PanoSpace\data", f"negative_sampling_DeepPanoContext_ReplicaPano.json")
    
    # 读取cognitive_map.json，获取图片中出现的对象
    with open(cog_map_path, 'r', encoding='utf-8') as f:
        cog_data = json.load(f)
    present_objs = set(cog_data.get("class_count", {}).keys())
    
    # 读取对应的negative_sampling文件
    with open(neg_file_path, 'r', encoding='utf-8') as f:
        neg_data = json.load(f)
    obj_freq = neg_data.get("object_frequency", {})
    cooccur_count = neg_data.get("cooccurrence_count", {})
    
    # 筛选object_frequency中未出现在图片中的对象，选出频率最高的n个
    missing_freq = {obj: freq for obj, freq in obj_freq.items() if obj not in present_objs}
    freq_candidates = sorted(missing_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # 从cooccurrence_count中，对于cognitive_map中出现的每个对象，找出共现值最高且未出现在图片中的对象
    cooccur_missing = {}
    for present in present_objs:
        inner = cooccur_count.get(present, {})
        for candidate, count in inner.items():
            if candidate in present_objs:
                continue
            if candidate not in cooccur_missing or count > cooccur_missing[candidate]:
                cooccur_missing[candidate] = count
    
    # 去除已在freq_candidates中出现的对象，防止重复
    freq_selected = set(obj for obj, _ in freq_candidates)
    cooccur_filtered = {obj: count for obj, count in cooccur_missing.items() if obj not in freq_selected}
    cooccur_candidates = sorted(cooccur_filtered.items(), key=lambda x: x[1], reverse=True)[:m]
    
    return freq_candidates, cooccur_candidates

if __name__ == "__main__":
    # 示例：以一个子文件夹路径为例，并设置n和m值
    folder = r"e:\00\PanoSpace\data\ReplicaPano-large_apartment_0_006-Scene_Info-00057"
    freq, cooccur = negative_sampling(folder, 5, 5)
    print("Top missing by frequency:", freq)
    print("Top missing by cooccurrence:", cooccur)
