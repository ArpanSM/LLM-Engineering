
import pandas as pd
from unsloth import FastLanguageModel
import torch

# load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "arpansm10/Llama-3.2-3B-Math-Reasoning",
    max_seq_length = 8192,
    dtype = None,
    load_in_4bit = True,
)

# get chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer,chat_template = "llama-3.1")

# load test data
test_data = pd.read_csv("gsm-hard-test-results.csv")

# load utils
from utils import call_litellm, EvaluationResult, SYSTEM_PROMPT

def generate_response(model, tokenizer, question):
  messages = [{"role": "user", "content": question}]
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize = True,
      add_generation_prompt = True, # Must add for generation
      return_tensors = "pt",
  ).to("cuda")

  outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True, temperature = 1)
  response = tokenizer.batch_decode(outputs)
  return response

# evaluate response
def evaluate_response(question: str, correct_answer: str, model_response: str) -> EvaluationResult:
    user_prompt = f"""Question: {question}\nCorrect Answer: {correct_answer}\nModel Response: {model_response}"""

    _, output = call_litellm(
        model="gemini/gemini-2.0-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=EvaluationResult,
        temperature=0,
    )
    return output

# Process test data
base_responses = []
base_scores = []

for idx, row in test_data.iterrows():
    question = row["input"]
    correct_answer = row["target"]

    # Generate base model response
    base_response = generate_response(model, tokenizer, question)
    print(f'Question: {question}')
    print(f'Correct Answer: {correct_answer}')
    print(f'Base Model Response: {base_response}')
    base_responses.append(base_response)

    # Evaluate response
    try:
        eval_result = evaluate_response(question, correct_answer, base_response)
        print(f'Evaluation Result: {eval_result}')
        base_scores.append(eval_result['score'])
    except Exception as e:
        print(f"Error evaluating row {idx}: {e}")
        base_scores.append(0)
    print(f'-----------------------------------------------------------------')

# Add results to dataframe
test_data["llama-3-2-3b-finetunedModelResponse"] = base_responses
test_data["llama-3-2-3b-finetunedModelScore"] = base_scores

# Save results
test_data.to_csv("gsm-hard-test-results.csv", index=False)

# Calculate accuracy
accuracy = test_data["llama-3-2-3b-finetunedModelScore"].mean()
print(f"Finetuned Model Accuracy: {accuracy:.2%}")
# Finetuned Model Accuracy: 46.00%
