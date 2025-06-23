from llama_cpp import Llama
import os

# === Path to your GGUF model ===
model_path = os.path.abspath("phi-2.Q4_K_M.gguf")  # Adjust path or filename as needed
model_path = os.path.abspath("mistral-7b-instruct-v0.2.Q4_K_M.gguf")  # or change path here

# === Load model ===
llm = Llama(
    model_path=model_path,
    n_ctx=512,     # Enough for simple prompts
    n_threads=4,   # Adjust for your CPU
    verbose=False
)

# === Prompt ===
prompt = "In which country is University of Edinburgh?"
print("🧠 Prompt:", prompt);
response = llm(prompt, stop=["\n"])
print("🧠 LLM Response:", response["choices"][0]["text"].strip())
