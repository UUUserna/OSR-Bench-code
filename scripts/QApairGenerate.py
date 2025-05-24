import negative_sampling
import numpy as np
import os
import json
import datetime


def find_object_positions(cognitive_map, object_name):
    positions = []
    for row_idx, row in enumerate(cognitive_map):
        for col_idx, cell in enumerate(row):
            if object_name in cell:
                positions.append((row_idx, col_idx))
    return positions


def find_closest_object(pos_obj_pos, obj_positions):
    min_dist = float("inf")
    closest_pos = None
    closest_angle = float("inf")
    for pos in obj_positions:
        dist = np.linalg.norm(np.array(pos) - pos_obj_pos)
        if dist < min_dist:
            min_dist = dist
            closest_pos = pos
            closest_angle = np.arctan2(
                pos[0] - pos_obj_pos[0], pos[1] - pos_obj_pos[1]
            ) % (2 * np.pi)
        elif dist == min_dist:
            angle = np.arctan2(pos[0] - pos_obj_pos[0], pos[1] - pos_obj_pos[1]) % (
                2 * np.pi
            )
            if angle < closest_angle:
                closest_angle = angle
                closest_pos = pos
    return np.array(closest_pos), min_dist


def find_closest_querying_object(cognitive_map, positioning_object, querying_objects):
    pos_obj_pos = find_object_positions(cognitive_map, positioning_object)
    if len(pos_obj_pos) != 1:
        raise ValueError("The positioning object is not found or is not unique in the picture.")
    pos_obj_pos = np.array(pos_obj_pos[0])

    closest_qry_obj = None
    closest_qry_pos = None
    min_distance = float("inf")

    for querying_object in querying_objects:
        qry_obj_positions = find_object_positions(cognitive_map, querying_object)
        if not qry_obj_positions:
            continue

        qry_obj_pos, qry_distance = find_closest_object(pos_obj_pos, qry_obj_positions)
        if qry_distance < min_distance:
            min_distance = qry_distance
            closest_qry_obj = querying_object
            closest_qry_pos = qry_obj_pos

    if closest_qry_obj is None:
        raise ValueError("None of the querying objects were found in the picture.")

    return closest_qry_obj, closest_qry_pos, min_distance


def query_object_position(
    cognitive_map, positioning_object, orienting_object, querying_object
):

    pos_obj_pos = find_object_positions(cognitive_map, positioning_object)
    if len(pos_obj_pos) != 1:
        raise ValueError("The positioning object is not found or is not unique in the picture.")
    pos_obj_pos = np.array(pos_obj_pos[0])

    ori_obj_positions = find_object_positions(cognitive_map, orienting_object)
    qry_obj_positions = find_object_positions(cognitive_map, querying_object)

    if not ori_obj_positions or not qry_obj_positions:
        raise ValueError("The orienting object or querying object is not found in the picture.")

    ori_obj_pos, ori_distance = find_closest_object(pos_obj_pos, ori_obj_positions)
    qry_obj_pos, qry_distance = find_closest_object(pos_obj_pos, qry_obj_positions)

    def classify_vector_direction(v1, v2):
        # 计算点积和向量模长
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 计算夹角的余弧度值，并确保在[-1,1]范围内
        cos_theta = dot / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # 夹角（弧度）
        theta_deg = np.degrees(theta)  # 转换为角度

        # 计算二维叉积（标量）
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # 判断前后
        if dot > 0:
            pos = "Front"  # 前方
        elif dot < 0:
            pos = "Back"  # 后方
        else:  # Orthogonal 正交
            pos = ""

        # 判断左侧还是右侧
        if cross > 0:
            side = "Left"
        elif cross < 0:
            side = "Right"
        else:
            side = ""

        # # 若正好正交，直接返回 Left/Right（依据叉积）或直接返回0°
        # if pos == "Orthogonal":
        #     return side if side else "Undefined", theta_deg

        direction = f"{pos} {side}".strip()  # 拼接，如 "Front Left" 或 "Back Right"
        return direction, theta_deg

    direction_vector = ori_obj_pos - pos_obj_pos

    relative_vector = qry_obj_pos - pos_obj_pos

    return classify_vector_direction(direction_vector, relative_vector)


def load_cognitive_map(folder_path):
    cog_map_path = os.path.join(folder_path, "cognitive_map.json")
    with open(cog_map_path, "r", encoding="utf-8") as f:
        cog_data = json.load(f)
    cognitive_map = cog_data.get("cognitive_map", [])
    class_count = cog_data.get("class_count", {})
    return cognitive_map, class_count


def object_counting(class_count, querying_object):
    count = class_count.get(querying_object, 0)
    question = f"How many {querying_object}(s) are in this room?"
    answer = str(count)
    return question, answer


def relative_distance(cognitive_map, category, choices):
    question = (
        f"Measuring from the closest point of each object, which of these objects ({', '.join(choices)}) "
        f"is the closest to the {category} in this PANORAMA?"
    )

    pos_obj_positions = find_object_positions(cognitive_map, category)
    if len(pos_obj_positions) == 0:
        answer = "The positioning object is not found in the picture."
        return question, answer
    if len(pos_obj_positions) > 1:
        answer = "The positioning object is not unique in the picture."
        return question, answer
    pos_obj_pos = np.array(pos_obj_positions[0])

    closest_choice = None
    min_distance = float("inf")

    for choice in choices:
        choice_positions = find_object_positions(cognitive_map, choice)
        if not choice_positions:
            continue

        choice_pos, choice_distance = find_closest_object(pos_obj_pos, choice_positions)
        if choice_distance < min_distance:
            min_distance = choice_distance
            closest_choice = choice

    if closest_choice is None:
        answer = "None of the candidates were found in the picture."
    else:
        answer = closest_choice

    return question, answer


def relative_direction(
    cognitive_map, positioning_object, orienting_object, querying_object
):
    question = (
        f"If I am standing by the {positioning_object} and facing the closest {orienting_object}, "
        f"is the closest {querying_object} to my front, back, left, right, front-left, front-right, back-left, or back-right in this PANORAMA?"
    )

    pos_obj_positions = find_object_positions(cognitive_map, positioning_object)
    if len(pos_obj_positions) == 0:
        answer = "The positioning object is not found in the picture."
        return question, answer
    if len(pos_obj_positions) > 1:
        answer = "The positioning object is not unique in the picture."
        return question, answer
    pos_obj_pos = np.array(pos_obj_positions[0])

    ori_obj_positions = find_object_positions(cognitive_map, orienting_object)
    qry_obj_positions = find_object_positions(cognitive_map, querying_object)

    if not ori_obj_positions or not qry_obj_positions:
        answer = "The orienting object or querying object is not found in the picture."
        return question, answer

    direction, _ = query_object_position(
        cognitive_map, positioning_object, orienting_object, querying_object
    )

    answer = direction
    return question, answer


def save_abnormal_qa(abnormal_qa_list):
    """保存异常QA对到json文件"""
    if not abnormal_qa_list:
        return
        
    output_path = r"e:\00\PanoSpace\abnormal_qa.json"
    data = {
        "abnormal_qa": abnormal_qa_list,
        "metadata": {
            "creation_date": datetime.date.today().isoformat(),
            "version": "1.0"
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_qa(cognitive_folder, qa_number):
    # 读取 cognitive_map.json 文件
    cognitive_map, class_count = load_cognitive_map(cognitive_folder)
    import random
    categories = list(class_count.keys())
    categories_str = ", ".join(categories)
    
    abnormal_qa_list = []  # 收集异常QA对
    base_name = os.path.basename(cognitive_folder)
    image_id = f"data/{base_name}/rgb.png"
    
    turn0_question = (
        "[Task] This PANORAMA captures an indoor scene. Your objective is to identify specific objects within the panorama, "
        "understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene "
        "is represented by a 10x10 grid. [Rule] 1. We provide the categories to care about in this scene: {" + categories_str + "}. "
        "Focus ONLY on these categories. 2. Estimate the center location of each instance within the provided categories, "
        "assuming the entire scene is represented by a 10x10 grid. 3. If a category contains multiple instances, include all of them. "
        "4. Each object’s estimated location should accurately reflect its real position in the scene, preserving the relative spatial "
        "relationships among all objects. 5. Consider object distance from camera when positioning them. Closer objects should be placed "
        "near the center of the grid, while distant objects should be placed toward the edges. 6. If an object is partially visible or occluded, "
        "estimate its full position based on the visible parts. [Output] Present the estimated center locations for each object as a list within a dictionary. "
        "STRICTLY follow this JSON format And use INTEGERS with 10 to represent the coordinates: {\"category name\": [[x_1, y_1], ...], ...}"
    )
    
    conversation = []
    conversation.append({
        "turn_id": 0,
        "question": turn0_question,
        "answer": cognitive_map,
        "skills_tested": "cognitive_map"
    })
    turn_id = 1

    # If there are fewer than 5 classes, simply generate one object_counting question per class.
    if len(categories) < 5:
        for cat in categories:
            q, a = object_counting(class_count, cat)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "object_counting"
            })
            turn_id += 1
    else:
        # Non-repeating sampling for object_counting questions.
        sampled_categories = random.sample(categories, min(qa_number, len(categories)))
        for cat in sampled_categories:
            q, a = object_counting(class_count, cat)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "object_counting"
            })
            turn_id += 1

        # 只有在类别数量足够时才生成相对距离和方向问题
        if len(categories) >= 5:
            # 相对距离问题生成
            pos_candidates = [c for c, cnt in class_count.items() if cnt == 1]
            attempts = 0
            successful_qa = 0
            while successful_qa < qa_number and attempts < qa_number * 2:
                attempts += 1
                if pos_candidates:
                    turn_category = random.choice(pos_candidates)
                else:
                    turn_category = random.choice(categories)
                other_choices = [c for c in categories if c != turn_category]
                if len(other_choices) >= 4:
                    turn_choices = random.sample(other_choices, 4)
                else:
                    turn_choices = other_choices
                
                q, a = relative_distance(cognitive_map, turn_category, turn_choices)
                # 检查是否为异常答案
                if any(x in a for x in ["not found", "not unique", "None of the candidates"]):
                    abnormal_qa_list.append({
                        "image_id": image_id,
                        "question": q,
                        "answer": a,
                        "skills_tested": "relative_distance"
                    })
                    continue
                
                conversation.append({
                    "turn_id": turn_id,
                    "question": q,
                    "answer": a,
                    "skills_tested": "relative_distance"
                })
                turn_id += 1
                successful_qa += 1

            # 相对方向问题生成
            attempts = 0
            successful_qa = 0
            while successful_qa < qa_number and attempts < qa_number * 2:
                attempts += 1
                if pos_candidates:
                    positioning_object = random.choice(pos_candidates)
                else:
                    positioning_object = random.choice(categories)
                remaining = [c for c in categories if c != positioning_object]
                orienting_object = random.choice(remaining)
                remaining.remove(orienting_object)
                querying_object = random.choice(remaining)
                
                q, a = relative_direction(cognitive_map, positioning_object, orienting_object, querying_object)
                # 检查是否为异常答案
                if any(x in a for x in ["not found", "not unique"]):
                    abnormal_qa_list.append({
                        "image_id": image_id,
                        "question": q,
                        "answer": a,
                        "skills_tested": "relative_direction"
                    })
                    continue
                    
                conversation.append({
                    "turn_id": turn_id,
                    "question": q,
                    "answer": a,
                    "skills_tested": "relative_direction"
                })
                turn_id += 1
                successful_qa += 1

    base_name = os.path.basename(cognitive_folder)
    image_id = f"data/{base_name}/rgb.png"
    scene_info = {
        "image_id": image_id,
        "pre_prompt": "",
        "conversation": conversation,
        "metadata": {
            "creation_date": datetime.date.today().isoformat(),
            "version": "1.0"
        }
    }
    
    output_json_path = os.path.join(rf"e:\00\PanoSpace\data\{base_name}", f"{base_name}_qa.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=2, ensure_ascii=False)

    # 生成没有 turn 0 的 JSON 文件
    conversation_without_turn0 = conversation[1:]
    scene_info_without_turn0 = {
        "image_id": image_id,
        "pre_prompt": "",
        "conversation": conversation_without_turn0,
        "metadata": {
            "creation_date": datetime.date.today().isoformat(),
            "version": "1.0"
        }
    }
    output_json_path_without_turn0 = os.path.join(rf"e:\00\PanoSpace\data\{base_name}", f"{base_name}_qa_without_cogmap.json")
    with open(output_json_path_without_turn0, "w", encoding="utf-8") as f:
        json.dump(scene_info_without_turn0, f, indent=2, ensure_ascii=False)

    # 保存异常QA对
    if abnormal_qa_list:
        save_abnormal_qa(abnormal_qa_list)


def generate_qa_with_negative_s(cognitive_folder, qa_number):
    # 读取 cognitive_map.json 文件与 class_count
    cognitive_map, class_count = load_cognitive_map(cognitive_folder)
    import random
    freq_candidates, cooccur_candidates = negative_sampling.negative_sampling(cognitive_folder, 5, 5)
    neg_objects = set([obj for obj, _ in freq_candidates] + [obj for obj, _ in cooccur_candidates])
    orig_categories = set(class_count.keys())
    all_categories = list(orig_categories.union(neg_objects))
    categories_str = ", ".join(all_categories)
    
    turn0_question = (
        "[Task] This PANORAMA captures an indoor scene. Your objective is to identify specific objects within the panorama, "
        "understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene "
        "is represented by a 10x10 grid. [Rule] 1. We provide the categories to care about in this scene: {" + categories_str + "}. "
        "Focus ONLY on these categories. 2. Estimate the center location of each instance within the provided categories, "
        "assuming the entire scene is represented by a 10x10 grid. 3. If a category contains multiple instances, include all of them. "
        "4. Each object’s estimated location should accurately reflect its real position in the scene, preserving the relative spatial "
        "relationships among all objects. 5. Consider object distance from camera when positioning them. Closer objects should be placed "
        "near the center of the grid, while distant objects should be placed toward the edges. 6. If an object is partially visible or occluded, "
        "estimate its full position based on the visible parts. [Output] Present the estimated center locations for each object as a list within a dictionary. "
        "STRICTLY follow this JSON format And use INTEGERS with 10 to represent the coordinates: {\"category name\": [[x_1, y_1], ...], ...}"
    )
    
    conversation = []
    conversation.append({
        "turn_id": 0,
        "question": turn0_question,
        "answer": cognitive_map,
        "skills_tested": "cognitive_map"
    })
    turn_id = 1

    # If there are fewer than 5 categories (by union), only generate object_counting questions.
    if len(class_count) < 5:
        sampled_categories = random.sample(all_categories, min(int(qa_number/2), len(all_categories)))
        for cat in sampled_categories:
            q, a = object_counting(class_count, cat)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "object_counting"
            })
            turn_id += 1
    else:
        sampled_categories = random.sample(all_categories, min(qa_number, len(all_categories)))
        for cat in sampled_categories:
            q, a = object_counting(class_count, cat)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "object_counting"
            })
            turn_id += 1

        pos_candidates = [c for c, cnt in class_count.items() if cnt == 1]
        for _ in range(qa_number):
            if pos_candidates:
                turn_category = random.choice(pos_candidates)
            else:
                turn_category = random.choice(all_categories)
            other_choices = [c for c in all_categories if c != turn_category]
            if len(other_choices) >= 4:
                turn_choices = random.sample(other_choices, 4)
            else:
                turn_choices = other_choices
            q, a = relative_distance(cognitive_map, turn_category, turn_choices)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "relative_distance"
            })
            turn_id += 1

        for _ in range(qa_number):
            if pos_candidates:
                positioning_object = random.choice(pos_candidates)
            else:
                positioning_object = random.choice(all_categories)
            remaining = [c for c in all_categories if c != positioning_object]
            orienting_object = random.choice(remaining)
            remaining.remove(orienting_object)
            querying_object = random.choice(remaining)
            q, a = relative_direction(cognitive_map, positioning_object, orienting_object, querying_object)
            conversation.append({
                "turn_id": turn_id,
                "question": q,
                "answer": a,
                "skills_tested": "relative_direction"
            })
            turn_id += 1

    base_name = os.path.basename(cognitive_folder)
    image_id = f"data/{base_name}/rgb.png"
    scene_info = {
        "image_id": image_id,
        "pre_prompt": "",
        "conversation": conversation,
        "metadata": {
            "creation_date": datetime.date.today().isoformat(),
            "version": "1.0"
        }
    }
    
    output_json_path = os.path.join(rf"e:\00\PanoSpace\data\{base_name}", f"{base_name}_qa_with_negative.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=2, ensure_ascii=False)

    # 生成没有 turn 0 的 JSON 文件
    conversation_without_turn0 = conversation[1:]
    scene_info_without_turn0 = {
        "image_id": image_id,
        "pre_prompt": "",
        "conversation": conversation_without_turn0,
        "metadata": {
            "creation_date": datetime.date.today().isoformat(),
            "version": "1.0"
        }
    }
    output_json_path_without_turn0 = os.path.join(rf"e:\00\PanoSpace\data\{base_name}", f"{base_name}_qa_with_negative_without_cogmap.json")
    with open(output_json_path_without_turn0, "w", encoding="utf-8") as f:
        json.dump(scene_info_without_turn0, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    base_folder = r"e:\00\PanoSpace\data"
    prefixes = ["ReplicaPano"] # ["DeepPanoContext", "ReplicaPano"]
    qa_number = 5 #10, 5

    for folder_name in os.listdir(base_folder):
        if any(folder_name.startswith(prefix) for prefix in prefixes):
            cognitive_folder = os.path.join(base_folder, folder_name)
            if os.path.isdir(cognitive_folder):
                generate_qa(cognitive_folder, qa_number=qa_number)
                # generate_qa_with_negative_s(cognitive_folder, qa_number=qa_number)
                print(f"Generated QA pairs for {cognitive_folder}")

