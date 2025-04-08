from math_reasoning.utils import call_litellm

def chat(model_name: str, message: str):
    response, content = call_litellm(
        model=model_name,
        messages=[{"role": "user", "content": message}]
    )
    return content

if __name__ == "__main__":
    print(chat("gemini/gemini-2.0-flash-lite", "What is the capital of France?"))
