import openai
import re
import json
from typing import List, Tuple, Dict, Any

# 针对每种技能类型的专用提示
# 修改提示模板，使用 raw string 并转义不需要格式化的花括号
META_PROMPT_OBJECT_COUNTING = r"""
You are evaluating an object counting answer. 

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Evaluation rules:
- The answer must be an integer (including 0)
- Correct if the model's integer exactly matches the ground truth
- Also correct if the model answer contains additional text but has the correct number

Please respond in JSON format:
{{
    "score": 0 or 1,  # 0 for incorrect, 1 for correct
    "confidence": "high" or "middle" or "low"  # Your confidence level
}}
"""

META_PROMPT_RELATIVE_DISTANCE = r"""
You are evaluating a relative distance answer.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Evaluation rules:
- Normal answers are object names
- Special case answers may be one of these error messages:
    * "The positioning object is not found in the picture."
    * "The positioning object is not unique in the picture."
    * "None of the candidates were found in the picture."
- When the ground truth is one of these special error messages, the model answer must match exactly
- When the ground truth is an object name, the model answer is correct if it contains the matching object name (case insensitive)

Please respond in JSON format:
{{
    "score": 0 or 1,  # 0 for incorrect, 1 for correct
    "confidence": "high" or "middle" or "low"  # Your confidence level
}}
"""

META_PROMPT_RELATIVE_DIRECTION = r"""
You are evaluating a relative direction answer.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Evaluation rules:
- Normal answers are direction words: "Front", "Back", "Left", "Right", "Front-Left", "Front-Right", "Back-Left", "Back-Right"
- Direction words are case insensitive, and hyphens are optional (e.g., "front left" equals "Front-Left")
- Special case answers may be one of these error messages:
    * "The positioning object is not found in the picture."
    * "The positioning object is not unique in the picture."
    * "The orienting object or querying object is not found in the picture."
- When the ground truth is one of these special error messages, the model answer must match exactly

Please respond in JSON format:
{{
    "score": 0 or 1,  # 0 for incorrect, 1 for correct
    "confidence": "high" or "middle" or "low"  # Your confidence level
}}
"""

def evaluate_with_openai(
    question: str, 
    ground_truth: str, 
    model_answer: str, 
    skill: str, 
    api_key: str
) -> Dict[str, Any]:
    """
    Evaluate a single model answer against ground truth using DeepSeek-V3 via OpenAI API.
    Uses different prompts based on skill type to save tokens.
    
    Parameters:
    - question: The question
    - ground_truth: The ground truth answer
    - model_answer: The model's answer
    - skill: Question type ('object_counting', 'relative_distance', 'relative_direction')
    - api_key: OpenAI API key
    
    Returns:
    - Evaluation result with score (0 or 1) and confidence level (high/middle/low)
    """
    # Process the model answer (extract from <answer> tags if present)
    if "<answer>" in model_answer and "</answer>" in model_answer:
        answer_match = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
        if answer_match:
            model_answer = answer_match.group(1).strip()
    
    # Select the appropriate prompt based on skill type
    if skill == "object_counting":
        prompt_template = META_PROMPT_OBJECT_COUNTING
    elif skill == "relative_distance":
        prompt_template = META_PROMPT_RELATIVE_DISTANCE
    elif skill == "relative_direction":
        prompt_template = META_PROMPT_RELATIVE_DIRECTION
    else:
        raise ValueError(f"Unknown skill type: {skill}")
    
    # Create the evaluation prompt
    formatted_prompt = prompt_template.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    
    try:
        # For OpenAI Python SDK v1.0.0+
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            'score': result.get("score", 0),
            'confidence': result.get("confidence", "low")
        }
    
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        # Return default result on error
        return {
            'score': 0,
            'confidence': 'low',
            'error': str(e)
        }


# Example usage
def main():
    api_key = ""
    
    # Test cases for each skill type
    test_cases = [
        {
            "question": "How many picture(s) are in this room?",
            "ground_truth": "1",
            "model_answer": "There is 2 picture in the room.",
            "skill": "object_counting"
        },
        {
            "question": "Which of these objects is the closest to the piano?",
            "ground_truth": "chair",
            "model_answer": "<think>I can see a piano with several objects nearby. The chair appears to be closest.</think><answer>chair</answer>",
            "skill": "relative_distance"
        },
        {
            "question": "If I am standing by the stool and facing the closest picture, is the closest door to my front, back, left, right, front-left, front-right, back-left, or back-right in this PANORAMA?",
            "ground_truth": "Front-Left",
            "model_answer": "front-right",
            "skill": "relative_direction"
        }
    ]
    
    # Evaluate each test case
    for case in test_cases:
        result = evaluate_with_openai(
            case["question"], 
            case["ground_truth"], 
            case["model_answer"], 
            case["skill"], 
            api_key
        )
        
        # Output result
        print(f"Question Type: {case['skill']}")
        print(f"Question: {case['question']}")
        print(f"Ground Truth: {case['ground_truth']} | Model Answer: {case['model_answer']}")
        print(f"Evaluation Result: {result}")
        print("---")


if __name__ == "__main__":
    main()