'''SGlang and ollama together, defeats it purpose. 

This is essentially useless code, but it serves as a demo of how to patch SGLang's OpenAI backend to handle reasoning models that return output in a 'reasoning' field instead of 'content'.


'''



import sglang as sgl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import openai
import json
import yaml
from pathlib import Path
from sglang.lang.backend import openai as sgl_openai

# --- Load configuration ---
CONFIG_PATH = Path(__file__).parent / "config" / "settings.yml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# --- Patch SGLang's OpenAI backend to handle reasoning models ---
def patched_openai_completion(client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs):
    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                ret = client.chat.completions.create(messages=prompt, **kwargs)
                if len(ret.choices) == 1:
                    # Handle reasoning models (qwen3.5, o1, etc.) - output is in 'reasoning' field
                    comp = ret.choices[0].message.content or getattr(ret.choices[0].message, 'reasoning', '') or ""
                else:
                    comp = [c.message.content or getattr(c.message, 'reasoning', '') or "" for c in ret.choices]
            else:
                ret = client.completions.create(prompt=prompt, **kwargs)
                if isinstance(prompt, (list, tuple)):
                    comp = [c.text for c in ret.choices]
                else:
                    comp = ret.choices[0].text
                    if len(ret.choices) > 1:
                        comp = [c.text for c in ret.choices]

            token_usage.prompt_tokens += ret.usage.prompt_tokens
            token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            import time
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"RuntimeError {e}.")
            raise e
    return comp

sgl_openai.openai_completion = patched_openai_completion

# --- SGLang wired to Ollama's OpenAI-compat endpoint ---
sgl.set_default_backend(
    sgl.OpenAI(
        model_name=config["ollama"]["model_name"],
        base_url=config["ollama"]["base_url"],
        api_key=config["ollama"]["api_key"]
    )
)

# --- SGLang program ---
@sgl.function
def generate_response(s, prompt, max_tokens=None, temperature=None):
    max_tokens = max_tokens or config["generation"]["max_tokens"]
    temperature = temperature if temperature is not None else config["generation"]["temperature"]
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=temperature))

# --- SGLang program for structured clinical analysis ---
@sgl.function
def analyze_clinical_text(s, text, task, temperature=0.1):
    s += sgl.system("""You are a clinical research assistant for academic researchers.
You always respond with valid JSON only. No preamble, no explanation, no markdown fences.
Just the raw JSON object.""")
    
    s += sgl.user(f"""Task: {task}

Text: {text}

Respond with exactly this JSON structure:
{{
  "answer": "your main response here",
  "confidence": "high or medium or low",
  "caveats": ["caveat 1", "caveat 2"],
  "requires_hitl": false
}}""")
    
    s += sgl.assistant(sgl.gen("result", max_tokens=1024, temperature=temperature))

# --- FastAPI app ---
app = FastAPI(title="SGLang + Ollama")

# --- Pydantic contracts ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = config["generation"]["max_tokens"]
    temperature: float = config["generation"]["temperature"]

class GenerateResponse(BaseModel):
    response: str

class AnalyzeRequest(BaseModel):
    text: str
    task: str = "Summarize the key clinical findings"

class AnalyzeResponse(BaseModel):
    result: dict
    raw: str

# --- Routes ---
@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        state = generate_response.run(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        # Use named gen key first, fall back to text parsing if empty
        response = state.get("response", "")
        if not response:
            response = state.text().split("ASSISTANT:")[-1].strip()
        return GenerateResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        state = analyze_clinical_text.run(
            text=request.text,
            task=request.task
        )
        
        raw = state["result"].strip()
        
        # Strip markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        
        parsed = json.loads(raw)
        return AnalyzeResponse(result=parsed, raw=raw)
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Model returned invalid JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["reload"]
    )
