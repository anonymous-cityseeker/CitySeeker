from src.mllm.chatgpt import ChatGPT4o


# S Models (7-16B)
class InternVL2_5_8B(ChatGPT4o):
    def __init__(self):
        super().__init__()

class llama3_llava_next_8b_hf(ChatGPT4o):
    def __init__(self):
        super().__init__()

class llava_onevision_qwen2_7b_si_hf(ChatGPT4o):
    def __init__(self):
        super().__init__()

class MiniCPM_V_2_6(ChatGPT4o):
    def __init__(self):
        super().__init__()

class Phi_3_5_vision_instruct(ChatGPT4o):
    def __init__(self):
        super().__init__()

class gpt_4o_mini(ChatGPT4o):
    def __init__(self):
        super().__init__()

class Qwen2_VL_7B_Instruct(ChatGPT4o):
    def __init__(self):
        super().__init__()

class Llama_3_2_11B_Vision(ChatGPT4o):
    def __init__(self):
        super().__init__()

# M Models (26-38B)
class InternVL2_5_26B(ChatGPT4o):
    def __init__(self):
        super().__init__()

class InternVL2_5_38B(ChatGPT4o):
    def __init__(self):
        super().__init__()

class deepseek_vl2(ChatGPT4o):
    def __init__(self):
        super().__init__()

# L Models (70-90B)

class Qwen2_VL_72B_Instruct(ChatGPT4o):
    def __init__(self):
        super().__init__()

class InternVL2_5_78B(ChatGPT4o):
    def __init__(self):
        super().__init__()

class llama3_2_90b(ChatGPT4o):
    def __init__(self):
        super().__init__()

class gemini_1_5_pro(ChatGPT4o):
    def __init__(self):
        super().__init__()

class claude_3_5_sonnet(ChatGPT4o):
    def __init__(self):
        super().__init__()

class MiniMax_01(ChatGPT4o):
    def __init__(self):
        super().__init__()
        
# Uncertain Models
class gemini_2_0_flash_exp(ChatGPT4o):
    def __init__(self):
        super().__init__()
