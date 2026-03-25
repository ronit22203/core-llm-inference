import sglang as sgl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- SGLang wired to Ollama's OpenAI-compat endpoint ---
sgl.set_default_backend(
    sgl.OpenAI(
        model_name="qwen3.5:2b",          # ← match whatever you have in ollama list
        base_url="http://localhost:11434/v1",
        api_key="ollama"           # Ollama ignores this, but the field is required
    )
)

# --- SGLang program ---
@sgl.function
def generate_response(s, prompt, max_tokens=256, temperature=0.7):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen(max_tokens=max_tokens, temperature=temperature))

# --- FastAPI app ---
app = FastAPI(title="SGLang + Ollama")

# --- Pydantic contracts ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str

# --- Routes ---
@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        state = generate_response.run(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return GenerateResponse(response=state.text().split("ASSISTANT:")[-1].strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # reload=True → hot reloads on file save, great for dev