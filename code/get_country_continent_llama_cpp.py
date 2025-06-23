import os
import pandas as pd
import time
from llama_cpp import Llama

# === Model Path ===
model_path = os.path.abspath("mistral-7b-instruct-v0.2.Q4_K_M.gguf")  # Update if needed

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
print(f"✅ Found model at {model_path}")

# === Initialize LLM ===
llm = Llama(
    model_path=model_path,
    model_type="mistral",   
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("✅ Mistral LLM loaded.")

# === Load University List ===
input_csv = "unique_universities.csv"
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"❌ Input file not found: {input_csv}")

df = pd.read_csv(input_csv)
universities = df["university"].dropna().tolist()
print(f"📚 Loaded {len(universities)} universities.")

# === Generate Country + Continent ===
results = []

for i, uni in enumerate(universities, 1):
    print(f"{i}/{len(universities)}: {uni}")

    prompt_country = f"Answer with only the country name. What country is {uni} located in?"
    response_country = llm(prompt_country, stop=["\n"])
    country = response_country["choices"][0]["text"].strip()

    prompt_continent = f"Answer with only the continent name. What continent is {uni} located in?"
    response_continent = llm(prompt_continent, stop=["\n"])
    continent = response_continent["choices"][0]["text"].strip()

    print(f"  → Country: {country}, Continent: {continent}")
    results.append([uni, country, continent])

    time.sleep(0.1)  # Small delay for stability

# === Save to CSV ===
output_csv = "universities_with_locations.csv"
df_out = pd.DataFrame(results, columns=["university", "country", "continent"])
df_out.to_csv(output_csv, index=False)
print(f"✅ Done. Results saved to {output_csv}")
