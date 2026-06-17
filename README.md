# Redrob Intelligent Candidate Discovery & Ranking Challenge
## Team: Rockers by Rahul
**Leader:** Ranadeep Saha

**Member of Google Developer Group**

### Project Overview
We have developed a robust, production-ready Proof of Concept (POC) that intelligently ranks 100,000 candidates for the "Senior AI Engineer — Founding Team" role. 

**Contact:** [ranadeep2021saha@gmail.com](mailto:ranadeep2021saha@gmail.com) | [LinkedIn](https://www.linkedin.com/in/ranadeep-saha-a03296404)

### How it Works
1. **Deep Job Understanding:** We analyze the JD to extract core requirements (embeddings, retrieval, production ML) and "vibe" signals (shipper attitude, responsiveness).
2. **Semantic Brain:** We use the `all-MiniLM-L6-v2` transformer model to convert candidate "career narratives" into 384-dimensional vectors. This allows the system to understand that a "Senior ML Engineer at Meesho" is a better fit than a "Marketing Manager" with AI keywords.
3. **Behavioral Trust Multiplier:** We integrate 23 signals (e.g., recruiter response rate, github activity, notice period) to calculate a "Trust Coefficient." This ensures we rank candidates who are not only skilled but also active and reachable.
4. **Honeypot Filtering:** A deterministic filter removes logically impossible profiles to ensure the high integrity of the shortlist.

### Performance
* **Runtime:** ~27 seconds for 100,000 candidates (CPU only).
* **Hardware:** 16GB RAM, No GPU required.

### How to Run
1. **Install Dependencies:**
   ```bash
   pip install sentence-transformers pandas numpy tqdm
   ```
2. **Phase 1: Pre-computation (One-time)**
   ```bash
   python precompute_embeddings.py
   ```
3. **Phase 2: Ranking**
   ```bash
   python rank.py
   ```
   The final output will be saved as `team_gemini.csv`.
