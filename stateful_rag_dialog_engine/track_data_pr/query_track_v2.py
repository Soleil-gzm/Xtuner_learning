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
    json_file: str, 
    iterations: int=20,
    k: int = 1000,
):
    """
    file_path: Excel文件路径
    json_file: 输出的JSON文件路径
    """
    xls = pd.ExcelFile(file_path)
    data = []
    
    for _ in range(iterations):
        for sheet_name in xls.sheet_names:
            # 读取当前sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # 替换换行符
            df = df.map(replace_newline)
            
            # 检查列数是否为4 
            if len(df.columns) != 4:
                print(f"Sheet '{sheet_name}' 不包含4列，跳过处理")
                continue
            
            df.columns  = ['ID', 'Assistant', 'User', 'Output']

            for _, row in df.iterrows(): 
                assistant_options = [opt.strip() for opt in str(row['Assistant']).split('/') if opt.strip()]
                user_options = [opt.strip() for opt in str(row['User']).split('/') if opt.strip()]
                print(f"Assistant options: {len(assistant_options)}")
                print(f"User options: {len(user_options)}")

                # 生成所有可能的组合
                all_combinations = list(itertools.product(assistant_options, user_options))
                
                print(f"Total combinations: {len(all_combinations)}")
                # 如果组合数大于需要的样本数，则进行采样
                if len(all_combinations) > k:
                    sampled_combinations = random.sample(all_combinations, k)
                else:
                    sampled_combinations = all_combinations
                
                # 处理采样后的组合
                for assistant_msg, user_msg in sampled_combinations:
                    history = [
                        {"role": "assistant", "message": assistant_msg},
                        {"role": "user", "message": user_msg}
                    ]
                    
                    system = load_and_format_prompt(prompt_path, history)
                    
                    data.append({ 
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
    print(len(data))

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data,  f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_sheet_to_txt('核实身份判定数据-v2.xlsx', './prompt_check.txt', 'output-check.json', iterations=1, k=800)
    # process_sheet_to_txt('还款意愿判定数据-v1.xlsx', './prompt_willing.txt', 'output-willing.json', iterations=1, k=500)
    # process_sheet_to_txt('核实信息意愿判定数据.xlsx', './prompt_verify.txt', 'output-verify.json', iterations=1, k=400)
    # process_sheet_to_txt('还款意愿判定数据-v2.xlsx', './prompt_willing.txt', 'output-willing.json', iterations=1, k=600)
    # process_sheet_to_txt('沟通意愿判定数据.xlsx', './prompt_continue.txt', 'output-continue.json', iterations=1, k=500)
