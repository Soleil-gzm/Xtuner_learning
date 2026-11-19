import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ================= 配置路径 =================
# 基础模型路径
base_model_path = "/home/zimeng/work/projects/Xtuner_learning/stateful_rag_dialog_engine/models/Qwen2.5-1.5B-Instruct"
# 刚刚转换出来的 Adapter 路径
adapter_path = "/home/zimeng/work/projects/Xtuner_learning/stateful_rag_dialog_engine/ft/test/hf_adapter_16390"
# 测试集路径
test_data_path = "/home/zimeng/work/projects/Xtuner_learning/stateful_rag_dialog_engine/track_data_pr/test-check.json"  # 改成你的测试集文件名

# ================= 1. 加载模型 =================
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 如果显存够（>16GB），可以直接用 fp16 加载。如果显存小，把下面的 quantization_config 打开
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    # quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
)

# 挂载 LoRA 权重
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("模型加载完毕！开始测试...\n")

# ================= 2. 辅助函数 =================
def extract_json(text):
    """从模型生成的各种花里胡哨的文本中提取 JSON"""
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return '{}'
        # 替换掉可能存在的 \_，防止解析失败
        return text[json_start:json_end].replace("\\_", "_")
    except:
        return '{}'

# ================= 3. 批量测试主循环 =================
def evaluate():
    with open(test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    json_failures = 0
    error_cases = []

    for i, item in enumerate(data):
        messages = item["messages"]
        
        # 提取系统 prompt、用户输入和真实标签
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        gt_assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "{}")
        
        try:
            ground_truth = json.loads(gt_assistant).get("node", "")
        except:
            continue # 如果训练数据的标签本身就是坏的，跳过

        # 构造 Qwen 标准对话格式
        chat = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # 推理生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=True, # 路由任务必须用贪心解码，保证输出稳定
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 解析预测结果
        pred_json_str = extract_json(response)
        try:
            pred_node = json.loads(pred_json_str).get("node", "解析失败")
            if pred_node == "解析失败":
                json_failures += 1
        except:
            pred_node = "解析失败"
            json_failures += 1
        
        # 比对结果
        if pred_node == ground_truth:
            correct += 1
        else:
            # 记录错误案例，方便后续 Bad Case 分析
            error_cases.append({
                "index": i,
                "ground_truth": ground_truth,
                "predicted": pred_node,
                "raw_output": response.strip()
            })

    # ================= 4. 打印最终报告 =================
    print("="*40)
    print(f"✅ 测试完成！总样本数: {total}")
    print(f"🎯 准确率 (Accuracy): {correct}/{total} = {correct/total*100:.2f}%")
    print(f"⚠️ JSON 解析失败数: {json_failures}")
    print("="*40)
    
    if error_cases:
        print("\n❌ 发现错误案例 (前 5 条):")
        for err in error_cases[:5]:
            print(f"索引: {err['index']} | 真实: {err['ground_truth']} | 预测: {err['predicted']}")
            print(f"原始输出: {err['raw_output']}\n")

if __name__ == "__main__":
    evaluate()