import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
CANDIDATES_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\india_challenge_extracted\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
VECTORS_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\candidate_vectors.npy"
IDS_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\candidate_ids.npy"
SUBMISSION_PATH = r"C:\Users\RANADEEP\Desktop\Gemini\team_gemini.csv"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Job Description Text (Extracted earlier)
JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI.
5–9 years experience. Production experience with embeddings-based retrieval systems, 
vector databases (Pinecone, Weaviate, etc.), Python, and evaluation frameworks (NDCG, MRR).
Scrappy product-engineering attitude. Based in Pune/Noida or willing to relocate from Tier-1 Indian cities.
Avoid keyword stuffers. Look for candidates who have shipped real ML systems.
"""

def calculate_trust_multiplier(signals):
    """
    Calculate a multiplier (0.5 to 1.5) based on behavioral signals.
    """
    if not signals:
        signals = {}
    multiplier = 1.0
    
    # Recruiter engagement
    response_rate = signals.get('recruiter_response_rate')
    if response_rate is None:
        response_rate = 0.5
    multiplier += (response_rate - 0.5) * 0.4 # +/- 0.2
    
    # Activity
    last_active = signals.get('last_active_date') or '2020-01-01'
    try:
        days_since_active = (datetime.now() - datetime.strptime(last_active, '%Y-%m-%d')).days
        if days_since_active < 30: multiplier += 0.1
        if days_since_active > 180: multiplier -= 0.2
    except: pass
    
    # Github signal
    github_score = signals.get('github_activity_score')
    if github_score is None:
        github_score = -1
    if github_score > 70: multiplier += 0.1
    
    # Notice period
    notice = signals.get('notice_period_days')
    if notice is None:
        notice = 60
    if notice <= 30: multiplier += 0.1
    if notice > 90: multiplier -= 0.1
    
    return max(0.5, min(1.5, multiplier))

def is_honeypot(candidate):
    """Detect impossible profiles."""
    signals = candidate.get('redrob_signals') or {}
    profile = candidate.get('profile') or {}
    
    # Rule 1: High experience with young age (simulated via signup/active gap or just high exp)
    exp = profile.get('years_of_experience')
    if exp is None:
        exp = 0
    if exp > 25: return True # Unlikely for "Senior AI Engineer" (5-9 target)
    
    # Rule 2: Expert in many skills but low duration
    skills_list = candidate.get('skills') or []
    for s in skills_list:
        if s and s.get('proficiency') == 'expert' and (s.get('duration_months') or 0) < 12:
            return True # Expert in < 1 year? Likely honeypot.
            
    # Rule 3: Logically impossible dates (simplified)
    # In real world, we'd check job dates vs total experience
    
    return False

def generate_reasoning(candidate, score):
    """Programmatic reasoning based on profile facts."""
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    
    name = profile.get('anonymized_name', 'Candidate')
    title = profile.get('current_title', 'Engineer')
    exp = profile.get('years_of_experience', 0)
    response_rate = signals.get('recruiter_response_rate', 0) * 100
    
    reason = f"{title} with {exp} years experience. "
    
    if score > 0.8:
        reason += f"Exceptional semantic match with production AI requirements. "
    elif score > 0.6:
        reason += f"Strong technical background matching key AI/ML skills. "
    else:
        reason += f"Relevant background in engineering with transferable ML interest. "
        
    if response_rate > 80:
        reason += f"Highly responsive (RR: {response_rate:.0f}%) and active on platform."
    elif signals.get('open_to_work_flag'):
        reason += "Marked as Open to Work; good availability."
        
    return reason[:250] # Limit length

def main():
    print("Loading pre-computed vectors...")
    candidate_vectors = np.load(VECTORS_PATH)
    candidate_ids = np.load(IDS_PATH)
    
    print("Encoding Job Description...")
    model = SentenceTransformer(MODEL_NAME)
    jd_vector = model.encode([JD_TEXT])[0]
    
    print("Calculating scores...")
    # 1. Cosine Similarity (Semantic Match)
    # Using dot product on normalized vectors
    norms = np.linalg.norm(candidate_vectors, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    semantic_scores = np.dot(candidate_vectors, jd_vector) / (norms * np.linalg.norm(jd_vector))
    
    # 2. Integrate Signals & Filter
    final_candidates = []
    
    print("Filtering and adjusting scores...")
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), total=100000):
            candidate = json.loads(line)
            cid = candidate['candidate_id']
            
            # Fast lookup of semantic score
            sem_score = semantic_scores[i]
            
            # Heuristic Filters
            if is_honeypot(candidate): continue
            
            # Location check (Pune/Noida/Hybrid/Relocate)
            signals = candidate.get('redrob_signals', {})
            location = candidate.get('profile', {}).get('location', '').lower()
            relocate = signals.get('willing_to_relocate', False)
            
            # Simple location boost/filter
            loc_multiplier = 1.0
            if any(city in location for city in ['pune', 'noida', 'delhi', 'ncr', 'gurgaon']):
                loc_multiplier = 1.1
            elif not relocate:
                loc_multiplier = 0.8 # Penalty if not local and not willing to relocate
            
            # Behavioral Multiplier
            trust_mult = calculate_trust_multiplier(signals)
            
            final_score = round(float(sem_score * trust_mult * loc_multiplier), 4)
            
            final_candidates.append({
                'candidate_id': cid,
                'score': final_score,
                'data': candidate # Keep for reasoning
            })
            
    # 3. Rank and select Top 100
    print("Ranking...")
    # Sort by score descending, then by candidate_id ascending for deterministic tie-breaking
    # Rounded score ensures we handle ties correctly as they will appear in the CSV
    ranked = sorted(final_candidates, key=lambda x: (-x['score'], x['candidate_id']))
    top_100 = ranked[:100]
    
    # 4. Generate Reasoning and Format CSV
    print("Generating submission file...")
    submission_rows = []
    for rank, item in enumerate(top_100, 1):
        reasoning = generate_reasoning(item['data'], item['score'])
        submission_rows.append({
            'candidate_id': item['candidate_id'],
            'rank': rank,
            'score': item['score'],
            'reasoning': reasoning
        })
        
    df = pd.DataFrame(submission_rows)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Success! Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
