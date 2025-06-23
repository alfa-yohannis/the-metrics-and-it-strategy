import os
import hashlib
from ctransformers import AutoModelForCausalLM

# === Configuration ===
MODEL_FILE = "phi-2.Q4_K_M.gguf"
EXPECTED_SHA256 = "324356668fa5ba9f4135de348447bb2bbe2467eaa1b8fcfb53719de62fbd2499"

# === Step 1: Check file exists ===
model_path = os.path.abspath(MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")
else:
    print(f"✅ Found model: {model_path}")

# === Step 2: Verify SHA256 checksum ===
def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print("🔍 Verifying SHA256...")
actual_sha = sha256sum(model_path)
print(f"  → Computed SHA256: {actual_sha}")
print(f"  → Expected SHA256: {EXPECTED_SHA256}")
if actual_sha != EXPECTED_SHA256:
    raise ValueError("❌ SHA256 hash mismatch — file may be corrupted.")
else:
    print("✅ SHA256 hash matches.")

# === Step 3: Load the model ===
print("🔄 Loading model using model_type='llama'...")
try:
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="phi",  # NOTE: Use "llama" even though it's phi
        gpu_layers=0,
        threads=4
    )
    print("✅ Model loaded successfully.")

    # Step 4: Test prompt
    prompt = "What country is Harvard University located in?"
    response = llm(prompt)
    print("✅ Model response:", response.strip())

except Exception as e:
    print("❌ Failed to load or run model:")
    print(e)
