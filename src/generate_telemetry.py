import os
import json
import csv
import random
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Setup path and load env variables
curr_dir = Path(__file__).resolve().parent
env_path = None
for _ in range(5):
    check_path = curr_dir / "ENV" / ".env"
    if check_path.exists():
        env_path = check_path
        break
    if curr_dir == curr_dir.parent:
        break
    curr_dir = curr_dir.parent

if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATASET_PATH = DATA_DIR / "telemetry_dataset.csv"

def generate_local_distribution(num_samples=5000):
    """
    Generates high-fidelity mock telemetry samples using programmatic distributions:
    Class 0 (Normal): Time spent is moderate to high, backspaces are proportional, low pastes.
    Class 1 (Shotgun Debugging): High run count, low time spent, higher syntax errors.
    Class 2 (Copy-Paste Dependency): High paste characters count, near zero backspaces.
    """
    print("[Telemetry Generator] Initializing programmatic distribution generator (Local Mode)...")
    samples = []
    
    # Class 0: Normal Coding
    num_class_0 = int(num_samples * 0.4)
    for _ in range(num_class_0):
        time_spent = max(15.0, np.random.normal(60.0, 15.0))
        run_count = max(1, int(np.random.poisson(2)))
        backspace_count = max(1, int(np.random.poisson(8)))
        paste_char_count = int(np.random.exponential(10.0))
        # Keep paste count moderate
        if paste_char_count > 30:
            paste_char_count = random.randint(5, 25)
        syntax_error_count = max(0, int(np.random.poisson(0.5)))
        samples.append([round(time_spent, 2), run_count, backspace_count, paste_char_count, syntax_error_count, 0])
        
    # Class 1: Shotgun Debugging
    num_class_1 = int(num_samples * 0.3)
    for _ in range(num_class_1):
        # High run count, low time spent
        time_spent = max(2.0, np.random.normal(10.0, 3.0))
        run_count = max(5, int(np.random.poisson(7)))
        backspace_count = max(0, int(np.random.poisson(2)))
        paste_char_count = int(np.random.exponential(5.0))
        if paste_char_count > 30:
            paste_char_count = random.randint(0, 15)
        syntax_error_count = max(1, int(np.random.poisson(3.5)))
        samples.append([round(time_spent, 2), run_count, backspace_count, paste_char_count, syntax_error_count, 1])

    # Class 2: Copy-Paster
    num_class_2 = num_samples - num_class_0 - num_class_1
    for _ in range(num_class_2):
        # High paste characters, low backspaces, low run counts initially
        time_spent = max(2.0, np.random.normal(8.0, 4.0))
        run_count = max(0, int(np.random.poisson(1)))
        backspace_count = max(0, int(np.random.binomial(2, 0.2))) # very low backspaces
        paste_char_count = max(31, int(np.random.normal(120.0, 40.0)))
        syntax_error_count = max(0, int(np.random.poisson(0.3)))
        samples.append([round(time_spent, 2), run_count, backspace_count, paste_char_count, syntax_error_count, 2])

    # Shuffle samples
    random.shuffle(samples)
    return samples

def query_openai_distributions():
    """Queries OpenAI API for behavioral distributions config"""
    print("[Telemetry Generator] Contacting OpenAI API for behavioral distributions config...")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    prompt = (
        "Generate a structured JSON configuration of probability distribution parameters for three classes of students: "
        "Class 0 (Normal): moderate-high time (mean 60s), low runs, high backspaces, low paste chars. "
        "Class 1 (Shotgun Debugging): low time (mean 12s), high runs (mean 8), low backspaces, moderate syntax errors. "
        "Class 2 (Copy-Paste Dependency): low time, low backspaces (mean 0), high paste chars (mean 150). "
        "Return only valid JSON containing mean and standard deviation for time_spent, run_count, backspace_count, paste_count, syntax_errors."
    )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    res_data = response.json()
    content = res_data["choices"][0]["message"]["content"]
    return json.loads(content)

def query_groq_distributions():
    """Queries Groq API for behavioral distributions config"""
    print("[Telemetry Generator] Contacting Groq API for behavioral distributions config...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    prompt = (
        "Generate a structured JSON configuration of probability distribution parameters for three classes of student behaviors. "
        "Return a JSON object containing keys: 'class_0', 'class_1', 'class_2'. Inside each, provide 'time_mean', 'time_std', 'runs_mean', 'backspace_mean', 'paste_mean', 'errors_mean'."
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a data science assistant that outputs JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    res_data = response.json()
    content = res_data["choices"][0]["message"]["content"]
    # Strip any potential markdown blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    return json.loads(content.strip())

def generate_dataset():
    config = None
    
    # 1. Try OpenAI
    if OPENAI_API_KEY:
        try:
            config = query_openai_distributions()
            print("[Telemetry Generator] OpenAI configurations loaded successfully.")
        except Exception as e:
            print(f"[Telemetry Generator] OpenAI Query failed: {e}. Attempting fallback to Groq...")
            
    # 2. Try Groq (Fallback 1)
    if not config and GROQ_API_KEY:
        try:
            config = query_groq_distributions()
            print("[Telemetry Generator] Groq configurations loaded successfully.")
        except Exception as e:
            print(f"[Telemetry Generator] Groq Query failed: {e}. Switching to programmatic distribution...")

    # 3. Process Config or Local sampling
    if config:
        print("[Telemetry Generator] Custom LLM-defined behavioral patterns detected. Sampling dataset...")
        # Fall back to programmatic mapping but using LLM specified averages
        samples = []
        try:
            # We parse the config parameters if they are matching. 
            # In case shape is unexpected, we fall back to standard local distribution
            c0 = config.get("class_0", config.get("Normal", {}))
            c1 = config.get("class_1", config.get("Shotgun", {}))
            c2 = config.get("class_2", config.get("Copy-Paste", {}))
            
            # Normal
            for _ in range(2000):
                time_s = max(10, np.random.normal(c0.get("time_mean", 60.0), c0.get("time_std", 15.0)))
                runs = max(0, int(np.random.poisson(c0.get("runs_mean", 2.0))))
                back = max(1, int(np.random.poisson(c0.get("backspace_mean", 8.0))))
                paste = max(0, int(np.random.poisson(c0.get("paste_mean", 5.0))))
                errors = max(0, int(np.random.poisson(c0.get("errors_mean", 0.5))))
                samples.append([round(time_s, 2), runs, back, paste, errors, 0])
                
            # Shotgun
            for _ in range(1500):
                time_s = max(1, np.random.normal(c1.get("time_mean", 12.0), c1.get("time_std", 4.0)))
                runs = max(4, int(np.random.poisson(c1.get("runs_mean", 7.0))))
                back = max(0, int(np.random.poisson(c1.get("backspace_mean", 1.5))))
                paste = max(0, int(np.random.poisson(c1.get("paste_mean", 8.0))))
                errors = max(1, int(np.random.poisson(c1.get("errors_mean", 3.0))))
                samples.append([round(time_s, 2), runs, back, paste, errors, 1])
                
            # Copy Paste
            for _ in range(1500):
                time_s = max(1, np.random.normal(c2.get("time_mean", 8.0), c2.get("time_std", 4.0)))
                runs = max(0, int(np.random.poisson(c2.get("runs_mean", 1.0))))
                back = max(0, int(np.random.poisson(c2.get("backspace_mean", 0.3))))
                paste = max(35, int(np.random.normal(c2.get("paste_mean", 120.0), 30.0)))
                errors = max(0, int(np.random.poisson(c2.get("errors_mean", 0.3))))
                samples.append([round(time_s, 2), runs, back, paste, errors, 2])
                
            random.shuffle(samples)
        except Exception as parse_err:
            print(f"[Telemetry Generator] LLM JSON parsing error: {parse_err}. Falling back to default distribution...")
            samples = generate_local_distribution()
    else:
        # Fallback to local distribution if API key not available or failed
        samples = generate_local_distribution()

    # Save to CSV
    print(f"[Telemetry Generator] Writing {len(samples)} records to local dataset at: {DATASET_PATH}")
    with open(DATASET_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_spent_sec", "run_count", "backspace_count", "paste_char_count", "syntax_error_count", "label"])
        writer.writerows(samples)
        
    print("[Telemetry Generator] Dataset generation complete!")

if __name__ == "__main__":
    generate_dataset()
