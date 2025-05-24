import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
import re

def extract_json_from_llm_output(llm_output):
    try:
        # 找到最后一个右花括号的位置
        end_idx = llm_output.rfind('}')
        
        if end_idx == -1:
            print("输入字符串中未找到JSON对象")
            return None
        
        # 从最后一个右花括号向前寻找匹配的左花括号
        bracket_count = 1  # 开始时已经找到一个右花括号
        start_idx = end_idx - 1
        
        while start_idx >= 0:
            if llm_output[start_idx] == '}':
                bracket_count += 1
            elif llm_output[start_idx] == '{':
                bracket_count -= 1
            
            if bracket_count == 0:
                break
                
            start_idx -= 1
        
        # 检查是否找到了匹配的左花括号
        if bracket_count > 0:
            print("未找到匹配的JSON对象")
            return None
        
        # 提取JSON字符串
        json_str = llm_output[start_idx:end_idx+1]
        
        # 将JSON字符串解析为Python字典
        json_obj = json.loads(json_str)
        return json_obj
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

def convert_model_output_to_cogmap(model_output, grid_size=10):
    """将模型输出转换为cogmap格式"""
    cogmap = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    
    for obj_name, positions in model_output.items():
        for pos in positions:
            x, y = pos
            if 0 <= x < grid_size and 0 <= y < grid_size:
                cogmap[y][x].append(obj_name)
    
    # 计算class_count
    class_count = {}
    for obj_name, positions in model_output.items():
        class_count[obj_name] = len(positions)
    
    return cogmap, class_count

def rotate_cogmap(cogmap, rotation):
    """旋转认知地图
    rotation: 0, 1, 2, 3 对应 0°, 90°, 180°, 270° (顺时针旋转)
    """
    grid_size = len(cogmap)
    rotated_map = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    
    for y in range(grid_size):
        for x in range(grid_size):
            original_objects = cogmap[y][x]
            
            if rotation == 0:  # 0°
                new_y, new_x = y, x
            elif rotation == 1:  # 90° 顺时针
                new_y, new_x = x, grid_size - 1 - y
            elif rotation == 2:  # 180°
                new_y, new_x = grid_size - 1 - y, grid_size - 1 - x
            elif rotation == 3:  # 270° 顺时针
                new_y, new_x = grid_size - 1 - x, y
            
            rotated_map[new_y][new_x] = original_objects.copy()

    return rotated_map

def extract_objects_with_positions(cogmap):
    """从认知地图中提取物体及其位置"""
    objects_by_class = {}
    grid_size = len(cogmap)
    
    for y in range(grid_size):
        for x in range(grid_size):
            for obj in cogmap[y][x]:
                if obj not in objects_by_class:
                    objects_by_class[obj] = []
                objects_by_class[obj].append((y, x))
    
    return objects_by_class

def calculate_distance_matrix(gt_positions, pred_positions):
    """计算两组位置之间的欧氏距离矩阵"""
    distance_matrix = np.zeros((len(gt_positions), len(pred_positions)))
    
    for i, gt_pos in enumerate(gt_positions):
        for j, pred_pos in enumerate(pred_positions):
            # 欧氏距离
            distance_matrix[i, j] = np.sqrt((gt_pos[0] - pred_pos[0])**2 + (gt_pos[1] - pred_pos[1])**2)
    
    return distance_matrix

def calculate_metrics_with_distance(gt_cogmap, pred_cogmap, distance_threshold=2.0):

    gt_objects = extract_objects_with_positions(gt_cogmap)
    pred_objects = extract_objects_with_positions(pred_cogmap)
    
    all_classes = set(list(gt_objects.keys()) + list(pred_objects.keys()))
    
    total_tp = 0
    total_fp = 0  #预测物体不在真实物体列表中，幻觉指标之一
    total_fn = 0
    average_distance = 0
    matched_count = 0
    
    # CHAIR指标相关变量
    hallucinated_classes = []  # 幻觉类别
    total_predicted_classes = len(pred_objects.keys())  # 预测的总类别数
    
    class_metrics = {}
    
    for obj_class in all_classes:
        gt_positions = gt_objects.get(obj_class, [])
        pred_positions = pred_objects.get(obj_class, [])
        
        # 检测幻觉类别：预测中有但真实中没有的类别
        if obj_class in pred_objects and obj_class not in gt_objects:
            hallucinated_classes.append(obj_class)
        
        # 初始假设：所有预测都是FP，所有真实都是FN
        fp_count = len(pred_positions)
        fn_count = len(gt_positions)
        tp_count = 0
        
        if not gt_positions or not pred_positions:
            # 该类别没有真实样本或预测样本
            class_metrics[obj_class] = {
                "precision": 0 if pred_positions else 1.0,  # 无预测时精确率定义为1
                "recall": 0 if gt_positions else 1.0,         # 无真实样本时召回率定义为1
                "f1": 0,
                "avg_distance": 15,
                "tp": 0,
                "fp": fp_count,
                "fn": fn_count
            }
            total_fp += fp_count
            total_fn += fn_count
            continue
            
        # 计算距离矩阵
        distance_matrix = calculate_distance_matrix(gt_positions, pred_positions)
        
        # 执行匈牙利算法
        gt_indices, pred_indices = linear_sum_assignment(distance_matrix)
        
        # 处理匹配结果
        sum_distance = 0
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            distance = distance_matrix[gt_idx, pred_idx]
            sum_distance += distance # Include all distances in sum
            
            # 距离阈值筛选 (only for TP/FP/FN counting)
            if distance <= distance_threshold:
              tp_count += 1
              # 每找到一个匹配，减少一个FP和一个FN
              fp_count -= 1
              fn_count -= 1
        
        # 更新该类别的最终指标
        avg_distance = sum_distance / len(gt_indices) if len(gt_indices) > 0 else 15
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[obj_class] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_distance": avg_distance,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count
        }
        
        # 累加到全局指标
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count
        if len(gt_indices) > 0:  # Changed condition to include all matches
            average_distance += sum_distance
            matched_count += len(gt_indices)  # Changed to count all matches
          
          # 计算整体指标
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_avg_distance = average_distance / matched_count if matched_count > 0 else 15
    
    # 计算CHAIR指标
    # CHAIR_I: 幻觉类别占预测类别的比例
    chair_i = len(hallucinated_classes) / total_predicted_classes if total_predicted_classes > 0 else 0
    
    # CHAIR_S: 在此场景中可简化为是否存在幻觉类别(1或0)
    chair_s = 1 if len(hallucinated_classes) > 0 else 0
    
    # 统计幻觉对象总数和预测对象总数
    hallucinated_instances_count = 0
    total_predicted_instances_count = 0
    
    for obj_class, positions in pred_objects.items():
        instances_count = len(positions)
        total_predicted_instances_count += instances_count
        
        if obj_class in hallucinated_classes:
            hallucinated_instances_count += instances_count
    
    # 计算实例级CHAIR
    chair_instance = hallucinated_instances_count / total_predicted_instances_count if total_predicted_instances_count > 0 else 0
    
    return {
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "avg_distance": overall_avg_distance,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "chair_s": chair_s,                           # 添加CHAIR_S指标
            "chair_i": chair_i,                           # 添加CHAIR_I指标
            "chair_instance": chair_instance,             # 添加实例级CHAIR指标
            "hallucinated_classes": hallucinated_classes  # 添加幻觉类别列表
        },
        "per_class": class_metrics
    }


def evaluate_with_rotations(gt_cogmap, first_turn_response, distance_threshold=2.0):
  # Define lowest score result
  lowest_score = {
    "overall": {
      "precision": 0,
      "recall": 0, 
      "f1": 0,
      "avg_distance": 15,
      "tp": 0,
      "fp": 0,
      "fn": 0,
      "chair_s": 0,
      "chair_i": 0,
      "chair_instance": 0,
      "hallucinated_classes": []
    },
    "per_class": {}
  }

  try:
    # Extract JSON 
    json_in_llm_output = extract_json_from_llm_output(first_turn_response)
    if json_in_llm_output is None:
      print("未能提取到有效的JSON数据")
      return lowest_score, 0

    # Convert to cogmap
    try:
      pred_cogmap, pre_class_count = convert_model_output_to_cogmap(json_in_llm_output)
    except:
      print("无法将JSON数据转换为cogmap")
      return lowest_score, 0

    best_metrics = None
    best_f1 = -1
    best_rotation = 0

    # Try each rotation
    for rotation in range(4):
      try:
        rotated_pred = rotate_cogmap(pred_cogmap, rotation)
        metrics = calculate_metrics_with_distance(gt_cogmap, rotated_pred, distance_threshold)

        if metrics["overall"]["f1"] > best_f1:
          best_f1 = metrics["overall"]["f1"] 
          best_metrics = metrics
          best_rotation = rotation * 90
      except:
        print(f"旋转{rotation*90}度计算指标时出错")
        continue

    # If no valid rotation found
    if best_metrics is None:
      return lowest_score, 0

    return best_metrics, best_rotation

  except Exception as e:
    print(f"评估过程出错: {str(e)}")
    return lowest_score, 0
    

def evaluation_re(ground_truth, model_answer, skill, use_think_prompt=False):
  """
  评估单个模型回答的正确性

  参数:
  - ground_truth: 标准答案 (字符串)
  - model_answer: 模型回答 (字符串)
  - skill: 问题类型 ('object_counting', 'relative_distance', 'relative_direction')
  - use_think_prompt: 是否使用think提示（带<answer>标签）

  返回:
  - object_counting返回0-1之间的浮点数表示准确率
  - 其他类型返回1表示正确，0表示错误
  """
  if use_think_prompt:
    answer_match = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
    if answer_match:
      model_answer = answer_match.group(1).strip()
    else:
      print("未找到<answer>标签，使用原始回答进行评估")

  model_answer = model_answer.strip().lower()
  ground_truth = ground_truth.strip().lower()

  if skill == 'object_counting':
    numbers = re.findall(r'\d+', model_answer)
    if not numbers:
      return 0  # 没有找到数字
    model_num = int(numbers[0])
    gt_num = int(ground_truth)
    if model_num == gt_num:
      return 1
    else:
      # 计算差异的百分比，差异越大分数越低
      max_val = max(model_num, gt_num)
      diff = abs(model_num - gt_num)
      return max(0, 1 - (diff / max_val))

  elif skill == 'relative_distance':
    return 1 if model_answer == ground_truth else 0

  elif skill == 'relative_direction':
    direction_mapping = {
      'front': 'front',
      'back': 'back',
      'left': 'left',
      'right': 'right',
      'front left': 'front-left',
      'front-left': 'front-left',
      'front right': 'front-right',
      'front-right': 'front-right',
      'back left': 'back-left',
      'back-left': 'back-left',
      'back right': 'back-right',
      'back-right': 'back-right'
    }

    normalized_ma = model_answer
    for key, value in direction_mapping.items():
      if model_answer == key:
        normalized_ma = value
        break

    normalized_gt = ground_truth
    for key, value in direction_mapping.items():
      if ground_truth == key:
        normalized_gt = value
        break

    return 1 if normalized_ma == normalized_gt else 0



if __name__ == "__main__":
    # 测试extract_json_from_llm_output函数
    test_str = 'Based on the panorama image of this indoor scene, I\'ll identify the specified objects and estimate their center locations using a 10x10 grid.\n\n```json\n{\n    "door": [[2, 5], [8, 5]],\n    "picture": [[1, 6]],\n    "piano": [[9, 7]],\n    "window": [],\n    "chair": [[3, 6], [7, 6]],\n    "carpet": [[3, 3], [8, 3]],\n    "sofa_chair": [[2, 6]],\n    "stool": [],\n    "table": [[5, 5]],\n    "floor_lamp": [[7, 7]]\n}\n```\n\nI\'ve identified:\n- Two doors (one on the left side and another on the right side of the panorama)\n- One picture on the left wall\n- A piano in the far right\n- Two chairs (one black chair on the right side and another visible partially)\n- Two carpets/rugs (blue patterned ones on the floor)\n- One sofa chair/armchair (blue one on the left)\n- A wooden table in the center\n- One floor lamp on the right side\n- No windows or stools visible in the image (though there appears to be a glass door/entrance which I\'ve classified as a door)'
    groud_truth = [
        [
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [
            "door"
          ],
          [
            "picture"
          ],
          [
            "piano"
          ],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [
            "window"
          ],
          [],
          [],
          [
            "chair"
          ],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [
            "carpet"
          ],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [],
          [],
          [],
          [],
          [],
          [
            "carpet"
          ],
          [],
          [],
          []
        ],
        [
          [
            "sofa_chair",
            "window"
          ],
          [],
          [],
          [
            "chair",
            "stool"
          ],
          [
            "table"
          ],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [],
          [
            "table"
          ],
          [],
          [
            "floor_lamp"
          ],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [],
          [
            "window"
          ],
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        [
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ]
      ]
    best_metrics, best_rotation = evaluate_with_rotations(groud_truth, test_str)
    print("Best Rotation:", best_rotation, "degrees")
    print("Best Metrics:", best_metrics)