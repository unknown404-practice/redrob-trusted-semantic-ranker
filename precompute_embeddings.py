import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Paths
CANDIDATES_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\india_challenge_extracted\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
OUTPUT_VECTORS_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\candidate_vectors.npy"
OUTPUT_IDS_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\candidate_ids.npy"
JD_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\india_challenge_extracted\job_description_extracted\word\document.xml" # Use the extracted text instead if needed

# Model choice: all-MiniLM-L6-v2 is very fast and has good semantic coverage
MODEL_NAME = 'all-MiniLM-L6-v2'

def extract_narrative(candidate):
    """Concatenate key fields into a single career narrative."""
    profile = candidate.get('profile', {})
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    
    # Career history
    history = []
    for job in candidate.get('career_history', []):
        history.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    
    full_history = " ".join(history)
    
    # Skills
    skills = ", ".join([s.get('name', '') for s in candidate.get('skills', [])])
    
    return f"{headline}. {summary}. Experience: {full_history}. Skills: {skills}"

def main():
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Use GPU if available (for pre-computation only)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    ids = []
    narratives = []
    
    print("Reading candidates...")
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=100000):
            candidate = json.loads(line)
            ids.append(candidate['candidate_id'])
            narratives.append(extract_narrative(candidate))
    
    print(f"Encoding {len(narratives)} candidates (this may take a few minutes)...")
    # Batch encoding for efficiency
    vectors = model.encode(narratives, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Saving vectors to {OUTPUT_VECTORS_PATH}...")
    np.save(OUTPUT_VECTORS_PATH, vectors)
    np.save(OUTPUT_IDS_PATH, np.array(ids))
    
    print("Pre-computation complete.")

if __name__ == "__main__":
    main()
