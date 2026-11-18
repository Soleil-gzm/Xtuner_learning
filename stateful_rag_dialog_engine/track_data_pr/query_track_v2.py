import json
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
import pandas as pd
import random
import itertools

class State(BaseModel):
    node: str

def load_and_format_prompt(prompt_path, history):
    """
    导入DST prompt
    history需要重新格式化
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    history_reformat = ""
    # 只判断最后一轮对话的内容
    for item in history[-2:]:
        role, message = item['role'], item['message']
        if role == 'assistant':
            history_reformat += "专员：" + message + '\n'
        else:
            history_reformat += "客户：" + message + '\n'

    prompt_template_load = PromptTemplate.from_template(prompt_template)
    prompt_case = prompt_template_load.format(
        json_format=State.model_json_schema(),
        history=history_reformat,
    )
    return prompt_case

def replace_newline(text):
    if isinstance(text, str):
        return text.replace('\n', ' ')  # 将换行符替换为空格
    return text

def process_sheet_to_txt(
    file_path: str,
    prompt_path: str, 
    train_json_file: str, 
    test_json_file: str,
    iterations: int=20,
    k: int = 1000,
    train_ratio: float = 0.9, # 默认 90% 训练，10% 测试
):
    xls = pd.ExcelFile(file_path)
    train_data = []
    test_data = []
    
    for _ in range(iterations):
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df = df.map(replace_newline)
            
            if len(df.columns) != 4:
                continue
            
            df.columns = ['ID', 'Assistant', 'User', 'Output']

            for _, row in df.iterrows(): 
                assistant_options = [opt.strip() for opt in str(row['Assistant']).split('/') if opt.strip()]
                user_options = [opt.strip() for opt in str(row['User']).split('/') if opt.strip()]

                all_combinations = list(itertools.product(assistant_options, user_options))
                
                # 1. 先打乱所有组合
                random.shuffle(all_combinations)
                
                # 2. 如果组合太多，先截断到 k 个
                if len(all_combinations) > k:
                    all_combinations = all_combinations[:k]
                
                # 3. 按比例切分
                split_idx = int(len(all_combinations) * train_ratio)
                train_combs = all_combinations[:split_idx]
                test_combs = all_combinations[split_idx:]

                # 4. 分别处理训练集和测试集
                for comb_list, data_list in [(train_combs, train_data), (test_combs, test_data)]:
                    for assistant_msg, user_msg in comb_list:
                        history = [
                            {"role": "assistant", "message": assistant_msg},
                            {"role": "user", "message": user_msg}
                        ]
                        system = load_and_format_prompt(prompt_path, history)
                        
                        data_list.append({ 
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": ""},
                                {
                                    "role": "assistant",
                                    "content": row['Output'].replace('\\n', '\n'),
                                    "loss": "True"
                                }
                            ]
                        })
                        
    print(f"训练集生成完毕，共 {len(train_data)} 条")
    print(f"测试集生成完毕，共 {len(test_data)} 条")

    # 分别保存为两个文件
    with open(train_json_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(test_json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 这里记得改一下传入的参数，现在需要传两个输出文件名
    process_sheet_to_txt(
        '核实身份判定数据-v2.xlsx', 
        './prompt_check.txt', 
        'train-check.json',    # 训练集
        'test-check.json',     # 测试集
        iterations=1, 
        k=800,
        train_ratio=0.9        # 90% 训练，10% 测试
    )

if __name__ == "__main__":
    import os
    
    # 动态获取当前脚本的绝对路径（沿用上一回合的方法，防止以后换路径运行报错）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    excel_file = os.path.join(current_dir, '核实身份判定数据-v2.xlsx')
    prompt_file = os.path.join(current_dir, './prompt_check.txt')
    train_out = os.path.join(current_dir, 'train-check.json')
    test_out = os.path.join(current_dir, 'test-check.json')

    # 这里必须把 train 和 test 两个输出路径都传进去
    process_sheet_to_txt(
        file_path=excel_file, 
        prompt_path=prompt_file, 
        train_json_file=train_out,
        test_json_file=test_out,
        iterations=1, 
        k=800,
        train_ratio=0.9  # 90% 训练，10% 测试
    )

    # 如果你还要跑其他的任务，也按照上面的格式补齐参数：
    # process_sheet_to_txt(
    #     file_path='还款意愿判定数据-v1.xlsx', 
    #     prompt_path='./prompt_willing.txt', 
    #     train_json_file='train-willing.json',
    #     test_json_file='test-willing.json',
    #     iterations=1, k=500
    # )
