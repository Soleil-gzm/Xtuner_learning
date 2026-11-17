def router_map_fn(example):
    msgs = example["messages"]
    system_text = ""
    user_text = ""
    assistant_text = ""
    
    # 遍历消息，提取 role 对应的 content
    for m in msgs:
        if m["role"] == "system":
            system_text = m["content"]
        elif m["role"] == "user":
            user_text = m["content"]
        elif m["role"] == "assistant":
            assistant_text = m["content"]
            
    # XTuner 期望的标准单轮对话格式
    return {
        "conversation": [
            {
                "system": system_text,
                "input": user_text if user_text else " ", # 防止空字符串导致某些处理报错
                "output": assistant_text
            }
        ]
    }