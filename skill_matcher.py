"""
Skill Matcher Module
Matches resume skills against job descriptions, identifies skill gaps, and ranks job roles.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import faiss
from typing import Dict, List, Tuple
from resume_parser import ResumeParser


class SkillMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the skill matcher with a sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.model = None
        
        self.job_descriptions = self._load_job_descriptions()
        self.jd_embeddings = self._embed_job_descriptions()
        self.faiss_index = self._build_faiss_index()
        self.resume_parser = ResumeParser()

    def _load_job_descriptions(self) -> Dict[str, str]:
        """Load sample job descriptions."""
        return {
            "Software Engineer": """
                - 5+ years of experience in Python, Java, or Go.
                - Strong understanding of data structures, algorithms, and software design.
                - Experience with cloud platforms like AWS or GCP.
                - Proficient with Docker, Kubernetes, and CI/CD pipelines.
                - Excellent problem-solving and communication skills.
            """,
            "AI Engineer": """
                - 3+ years of experience in machine learning and deep learning.
                - Proficient in Python with libraries like TensorFlow, PyTorch, and Scikit-learn.
                - Experience with MLOps tools such as MLFlow and Kubeflow.
                - Strong background in model deployment, optimization, and monitoring.
                - Familiarity with NLP, computer vision, and reinforcement learning.
            """,
            "Data Analyst": """
                - 2+ years of experience in data analysis and visualization.
                - Proficient in SQL and Python (Pandas, NumPy).
                - Experience with BI tools like Tableau or Power BI.
                - Strong analytical and statistical skills.
                - Excellent communication and presentation skills.
            """,
            "Frontend Developer": """
                - 3+ years of experience in JavaScript, HTML, and CSS.
                - Proficient in React, Angular, or Vue.js frameworks.
                - Experience with responsive design and cross-browser compatibility.
                - Knowledge of version control systems like Git.
                - Strong attention to detail and user experience.
            """,
            "DevOps Engineer": """
                - 4+ years of experience in cloud infrastructure and automation.
                - Proficient in Docker, Kubernetes, and container orchestration.
                - Experience with CI/CD pipelines and infrastructure as code.
                - Knowledge of monitoring and logging tools.
                - Strong problem-solving and collaboration skills.
            """
        }

    def _embed_job_descriptions(self) -> Dict[str, np.ndarray]:
        """Create embeddings for job descriptions."""
        if not self.model:
            return {}
        
        embeddings = {}
        for role, description in self.job_descriptions.items():
            embeddings[role] = self.model.encode(description)
        return embeddings

    def _build_faiss_index(self) -> faiss.IndexFlatL2:
        """Build a FAISS index for efficient similarity search."""
        if not self.jd_embeddings:
            return None
        
        dimension = list(self.jd_embeddings.values())[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to the index
        for embedding in self.jd_embeddings.values():
            index.add(np.array([embedding]))
        
        return index

    def match_resume_to_roles(self, resume_skills: List[str]) -> List[Tuple[str, float]]:
        """Match resume skills against job roles and rank them."""
        if not self.model or not resume_skills:
            return []
        
        # Create embedding for resume skills
        resume_text = ' '.join(resume_skills)
        resume_embedding = self.model.encode(resume_text)
        
        # Calculate cosine similarity
        similarities = []
        for role, jd_embedding in self.jd_embeddings.items():
            similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
            similarities.append((role, float(similarity)))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def identify_skill_gaps(self, resume_skills: List[str], target_role: str) -> List[str]:
        """Identify skill gaps for a target job role."""
        if target_role not in self.job_descriptions:
            return ["Target role not found."]
        
        # Extract required skills from JD
        jd_text = self.job_descriptions[target_role]
        required_skills = self.resume_parser.extract_skills(jd_text)
        
        # Find missing skills
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        skill_gaps = [
            skill for skill in required_skills 
            if skill.lower() not in resume_skills_lower
        ]
        
        return skill_gaps

    def find_similar_skills_with_faiss(self, query_skill: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar skills using FAISS index."""
        if not self.faiss_index or not self.model:
            return []
        
        query_embedding = self.model.encode([query_skill])
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Map indices back to roles
        roles = list(self.job_descriptions.keys())
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(roles):
                results.append((roles[i], float(dist)))
            
        return results


# Example usage and testing
if __name__ == "__main__":
    matcher = SkillMatcher()
    
    # Sample resume skills
    sample_skills = ["Python", "SQL", "Pandas", "Data Analysis", "Communication"]
    
    # Match to roles
    ranked_roles = matcher.match_resume_to_roles(sample_skills)
    print("Ranked Job Roles:", ranked_roles)
    
    # Identify skill gaps for a target role
    target = "Data Analyst"
    gaps = matcher.identify_skill_gaps(sample_skills, target)
    print(f"Skill Gaps for {target}:", gaps)
    
    # Find similar skills/roles
    similar = matcher.find_similar_skills_with_faiss("Machine Learning")
    print("Similar Roles/Skills to 'Machine Learning':", similar)
from typing import Dict, List, Tuple
import json
import os

class SkillMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the skill matcher with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.job_descriptions = self._load_job_descriptions()
        self.jd_embeddings = self._create_jd_embeddings()
        self.faiss_index = self._build_faiss_index(self.jd_embeddings)
    
    def _load_job_descriptions(self) -> Dict[str, str]:
        """Load sample job descriptions from a JSON file."""
        # In a real application, this would come from a database or API
        sample_jds = {
            "Software Engineer": """
                - 5+ years of experience in Python, Java, or C++
                - Strong understanding of data structures and algorithms
                - Experience with cloud platforms like AWS or Azure
                - Familiarity with CI/CD pipelines and Docker
                - Excellent problem-solving and communication skills
            """,
            "AI Engineer": """
                - Proficient in Python with experience in TensorFlow or PyTorch
                - Strong background in machine learning, deep learning, and NLP
                - Experience with MLOps tools like MLFlow and Kubeflow
                - Knowledge of model deployment and optimization
                - Experience with big data technologies like Spark
            """,
            "Data Analyst": """
                - Expertise in SQL and data visualization tools like Tableau or Power BI
                - Strong analytical skills and experience with statistical analysis
                - Proficient in Python or R for data manipulation (pandas, numpy)
                - Experience with data warehousing and ETL processes
                - Excellent communication and presentation skills
            """
        }
        
        # Create a file to store JDs if it doesn't exist
        if not os.path.exists('job_descriptions.json'):
            with open('job_descriptions.json', 'w') as f:
                json.dump(sample_jds, f, indent=2)
        
        with open('job_descriptions.json', 'r') as f:
            return json.load(f)
    
    def _create_jd_embeddings(self) -> np.ndarray:
        """Create sentence embeddings for all job descriptions."""
        jd_texts = list(self.job_descriptions.values())
        return self.model.encode(jd_texts, convert_to_tensor=True).cpu().numpy()
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build a FAISS index for efficient similarity search."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def match_resume_to_jds_tfidf(self, resume_skills: List[str]) -> Dict[str, float]:
        """Match resume skills to JDs using TF-IDF and cosine similarity."""
        resume_skill_text = ' '.join(resume_skills)
        jd_texts = list(self.job_descriptions.values())
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_skill_text] + jd_texts)
        
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        scores = {}
        for i, role in enumerate(self.job_descriptions.keys()):
            scores[role] = cosine_similarities[i]
            
        return scores
    
    def match_resume_to_jds_semantic(self, resume_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Match resume to JDs using semantic search with FAISS."""
        resume_embedding = self.model.encode([resume_text], convert_to_tensor=True).cpu().numpy()
        
        distances, indices = self.faiss_index.search(resume_embedding, top_k)
        
        results = []
        roles = list(self.job_descriptions.keys())
        for i, idx in enumerate(indices[0]):
            similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
            results.append((roles[idx], similarity))
            
        return results
    
    def identify_skill_gaps(self, resume_skills: List[str], target_role: str) -> Dict[str, List[str]]:
        """Identify skill gaps for a target job role."""
        if target_role not in self.job_descriptions:
            return {"error": "Target role not found"}
        
        jd_text = self.job_descriptions[target_role]
        
        # Extract required skills from JD (simple version)
        # In a real app, this would use a more sophisticated skill extractor
        required_skills = [skill.strip() for skill in jd_text.lower().split('\n') if '-' in skill]
        required_skills = [re.sub(r'[^a-zA-Z0-9\s+#-]', '', s).replace('-', '').strip() for s in required_skills]
        
        # Normalize resume skills
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        # Find missing skills
        missing_skills = []
        for req_skill in required_skills:
            if not any(resume_skill in req_skill for resume_skill in resume_skills_lower):
                missing_skills.append(req_skill.title())
        
        return {
            "target_role": target_role,
            "skill_gaps": missing_skills,
            "current_skills": resume_skills
        }

# Example usage and testing
if __name__ == "__main__":
    matcher = SkillMatcher()
    
    # Sample resume data
    sample_resume_skills = ["Python", "Java", "AWS", "Docker", "Problem Solving"]
    sample_resume_text = "Experienced software engineer with a background in cloud computing and backend development."
    
    # Test TF-IDF matching
    tfidf_scores = matcher.match_resume_to_jds_tfidf(sample_resume_skills)
    print("TF-IDF Matching Scores:", tfidf_scores)
    
    # Test semantic matching
    semantic_matches = matcher.match_resume_to_jds_semantic(sample_resume_text)
    print("Semantic Matching Results:", semantic_matches)
    
    # Test skill gap analysis
    target_role = "Software Engineer"
    skill_gaps = matcher.identify_skill_gaps(sample_resume_skills, target_role)
    print(f"Skill Gaps for {target_role}:", json.dumps(skill_gaps, indent=2))
    
    target_role_ai = "AI Engineer"
    skill_gaps_ai = matcher.identify_skill_gaps(sample_resume_skills, target_role_ai)
    print(f"Skill Gaps for {target_role_ai}:", json.dumps(skill_gaps_ai, indent=2))
