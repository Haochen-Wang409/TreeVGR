import ast
import re
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from tqdm import tqdm
import torch
import numpy as np
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI


def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []
    
    for match in matches:
        box = match.strip()
        
        coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
        coord_match = re.match(coord_pattern, box)
        
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            
            if x1 < x2 and y1 < y2:
                # all_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                all_boxes.append([x1, y1, x2, y2])

    def calculate_average_iou(pred_boxes, target_boxes):
        def compute_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_width = max(0, inter_x_max - inter_x_min)
            inter_height = max(0, inter_y_max - inter_y_min)
            inter_area = inter_width * inter_height

            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)

            union_area = area1 + area2 - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        pred_coords = pred_boxes
        target_coords = target_boxes # x1,y1,x2,y2

        total_iou = 0.0
        num_targets = len(target_boxes)

        if num_targets == 0:
            return 0.0

        for t_coord in target_coords:
            best_iou = 0.0
            for p_coord in pred_coords:
                iou = compute_iou(t_coord, p_coord)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou

        return total_iou / num_targets

    return calculate_average_iou(all_boxes, target_boxes)


def eval_model_row(item):
    if item["category"] == "OCR":
        qs = item["question"]
    else:
        qs = item["question"] + " Options:\n" + item["multi-choice options"]

    content = [
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{item['image']}",
        },
        {
            "type": "text",
            "text": qs + "\nSelect the best answer to the above multiple-choice question based on the image. After the reasoning process, respond with only the letter of the correct option between <answer> and </answer>.",
        },
    ]

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """A conversation between user and assistant. The user asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format.""",
            }],
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += "<think>"

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
            max_new_tokens=1024,
            use_cache=True,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    box_iou = compute_box_iou(output_text[0], ast.literal_eval(item["target_instances"]))

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, output_text[0], re.DOTALL)
    ans = match.group(1).strip().upper() if match else output_text[0]

    item["prediction"] = ans
    item["iou"] = box_iou

    return item


# default model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "HaochenWang/TreeVGR-7B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True,
)

# default processor
processor = AutoProcessor.from_pretrained(
    "HaochenWang/TreeVGR-7B",
    min_pixels=1280*28*28, max_pixels=16384*28*28,
)


if __name__ == "__main__":
    # load data
    df = load_dataset("HaochenWang/TreeBench", data_files="TreeBench.tsv", delimiter="\t")["train"]

    # obtain results
    data = []
    pool = multiprocessing.Pool(processes=torch.cuda.device_count())
    with tqdm(total=len(df), desc="Processing") as pbar:
        for result in pool.imap(eval_model_row, df):
            if result is not None:
                data.append(result)
                pbar.update(1)

    pool.close()
    pool.join()
        
    results = {}
    tags = ["Perception/Attributes", "Perception/Material", "Perception/Physical State", 
            "Perception/Object Retrieval", "Perception/OCR",
            "Reasoning/Perspective Transform", "Reasoning/Ordering", "Reasoning/Contact and Occlusion",
            "Reasoning/Spatial Containment", "Reasoning/Comparison"]
    total = 0
    correct = 0
        
    for tag in tags:
        results[tag] = {"correct": 0, "total": 0}
        for item in data:
            if tag == item["category"]:
                total += 1
                results[tag]["total"] += 1
                # exact matching
                if item["prediction"].upper() == item["answer"].upper():
                    results[tag]["correct"] += 1
                    correct += 1

        acc = results[tag]["correct"] / results[tag]["total"]
        print(tag, f"{results[tag]['correct']}/{results[tag]['total']}={round(acc * 100, 2)}")
    print("==> Overall", f"{correct}/{total}={round(correct / total * 100, 2)}")

    iou = np.array([x["iou"] for x in data])
    print("==> Mean IoU:", round(np.mean(iou) * 100, 2))