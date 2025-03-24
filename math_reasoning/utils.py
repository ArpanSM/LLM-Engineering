import os
import json
import litellm
from litellm import Router, completion, completion_cost
from pydantic import BaseModel, Field
from typing import Optional

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# evaluation result
class EvaluationResult(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of evaluation within 20 words")
    actual_answer: str = Field(..., description="Actual answer from the question")
    extracted_answer: Optional[float] = Field(None, description="Numerical answer extracted from response")
    score: int = Field(..., description="1 if correct, 0 if incorrect")

# system prompt
SYSTEM_PROMPT = """
You are a math answer evaluator. You are given a question, correct answer and model response.
Compare the model's response with the correct answer.
Understand the LLM response and extract the final numerical answer from the model response.
Ignore differences in punctuation, spacing, or capitalization. Return score=1 only if the numerical values match exactly.
"""

router = Router(
    routing_strategy="usage-based-routing-v2",
    enable_pre_call_checks=True,
    model_list=[
        {
            "model_name": "gemini/gemini-2.0-flash",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash-lite",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash-lite",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "groq/llama-3.3-70b-versatile",
            "litellm_params":
            {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY_1")
            }
        },
        {
            "model_name": "groq/qwen-qwq-32b",
            "litellm_params":
            {
                "model": "groq/qwen-qwq-32b",
                "api_key": os.getenv("GROQ_API_KEY_1")
            }
        }
    ]
)

def call_litellm(**kwargs):
    response = router.completion(**kwargs)
    output = response.choices[0].message.content
    if kwargs.get('response_format'):
        output = json.loads(output)
    return response, output