"""
Resume Parser Module
Extracts structured information from PDF resumes including skills, experience, and education.
"""

import re
import fitz  # PyMuPDF
import spacy
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import json


class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with NLP model and skill ontology."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load skill ontology
        self.skill_ontology = self._load_skill_ontology()
        
        # Compile regex patterns for extraction
        self._compile_patterns()
    
    def _load_skill_ontology(self) -> Dict:
        """Load comprehensive skill taxonomy."""
        return {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "sql",
                "html", "css", "bash", "powershell", "perl", "lua", "dart"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "django", "flask", "fastapi", "spring",
                "express", "nodejs", "tensorflow", "pytorch", "keras", "scikit-learn",
                "pandas", "numpy", "matplotlib", "seaborn", "plotly", "bootstrap",
                "jquery", "redux", "next.js", "nuxt.js", "laravel", "symfony"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
                "oracle", "sql server", "cassandra", "dynamodb", "neo4j", "firebase"
            ],
            "cloud_platforms": [
                "aws", "azure", "gcp", "google cloud", "digital ocean", "heroku",
                "vercel", "netlify", "cloudflare", "ibm cloud"
            ],
            "devops_tools": [
                "docker", "kubernetes", "jenkins", "gitlab ci", "github actions",
                "terraform", "ansible", "vagrant", "helm", "prometheus", "grafana",
                "elk stack", "nginx", "apache", "ci/cd", "microservices"
            ],
            "ml_ai_tools": [
                "mlflow", "kubeflow", "airflow", "jupyter", "anaconda", "hadoop",
                "spark", "kafka", "mlops", "model deployment", "feature engineering",
                "deep learning", "computer vision", "nlp", "reinforcement learning"
            ],
            "soft_skills": [
                "leadership", "communication", "problem solving", "teamwork",
                "project management", "agile", "scrum", "critical thinking",
                "adaptability", "creativity", "time management", "analytical thinking"
            ],
            "tools_technologies": [
                "git", "github", "gitlab", "bitbucket", "jira", "confluence",
                "slack", "trello", "notion", "figma", "adobe creative suite",
                "vs code", "intellij", "pycharm", "vim", "emacs"
            ]
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for information extraction."""
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone pattern
        self.phone_pattern = re.compile(r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        
        # LinkedIn pattern
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/([a-zA-Z0-9-]+)')
        
        # GitHub pattern
        self.github_pattern = re.compile(r'github\.com/([a-zA-Z0-9-]+)')
        
        # Date patterns for experience
        self.date_patterns = [
            re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}/\d{4}\b'),
            re.compile(r'\b\d{4}\b'),
            re.compile(r'\b(Present|Current|Now)\b', re.IGNORECASE)
        ]
        
        # Section headers
        self.section_patterns = {
            'experience': re.compile(r'\b(experience|work history|employment|career|professional)\b', re.IGNORECASE),
            'education': re.compile(r'\b(education|academic|degree|university|college)\b', re.IGNORECASE),
            'skills': re.compile(r'\b(skills|technical skills|competencies|expertise)\b', re.IGNORECASE),
            'projects': re.compile(r'\b(projects|portfolio|work samples)\b', re.IGNORECASE),
            'certifications': re.compile(r'\b(certifications|certificates|credentials)\b', re.IGNORECASE)
        }
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from resume text."""
        contact_info = {}
        
        # Extract email
        email_match = self.email_pattern.search(text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Extract phone
        phone_match = self.phone_pattern.search(text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # Extract LinkedIn
        linkedin_match = self.linkedin_pattern.search(text)
        if linkedin_match:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # Extract GitHub
        github_match = self.github_pattern.search(text)
        if github_match:
            contact_info['github'] = f"github.com/{github_match.group(1)}"
        
        # Extract name using NLP
        if self.nlp:
            doc = self.nlp(text[:500])  # First 500 chars likely contain name
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    contact_info['name'] = ent.text
                    break
        
        return contact_info
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract and normalize skills from resume text."""
        skills = set()
        text_lower = text.lower()
        
        # Extract skills from all categories
        for category, skill_list in self.skill_ontology.items():
            for skill in skill_list:
                # Check for exact matches and variations
                if skill.lower() in text_lower:
                    skills.add(skill.title())
                
                # Check for skill variations (e.g., "js" for "javascript")
                skill_variations = self._get_skill_variations(skill)
                for variation in skill_variations:
                    if variation.lower() in text_lower:
                        skills.add(skill.title())
        
        return list(skills)
    
    def _get_skill_variations(self, skill: str) -> List[str]:
        """Get common variations of a skill."""
        variations = {
            'javascript': ['js', 'node.js', 'nodejs'],
            'python': ['py'],
            'machine learning': ['ml', 'ai', 'artificial intelligence'],
            'natural language processing': ['nlp'],
            'computer vision': ['cv'],
            'amazon web services': ['aws'],
            'google cloud platform': ['gcp'],
            'microsoft azure': ['azure'],
            'continuous integration': ['ci'],
            'continuous deployment': ['cd'],
            'version control': ['git', 'svn'],
            'database': ['db'],
            'user interface': ['ui'],
            'user experience': ['ux'],
            'application programming interface': ['api'],
            'software development kit': ['sdk'],
            'integrated development environment': ['ide']
        }
        return variations.get(skill.lower(), [])
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text."""
        experiences = []
        
        # Find experience section
        lines = text.split('\n')
        in_experience_section = False
        current_experience = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if we're entering experience section
            if self.section_patterns['experience'].search(line):
                in_experience_section = True
                continue
            
            # Check if we're leaving experience section
            if in_experience_section and any(pattern.search(line) for pattern in self.section_patterns.values() if pattern != self.section_patterns['experience']):
                if current_experience:
                    experiences.append(current_experience)
                    current_experience = {}
                in_experience_section = False
                continue
            
            if in_experience_section:
                # Extract dates
                dates = self._extract_dates(line)
                if dates:
                    if current_experience:
                        experiences.append(current_experience)
                    current_experience = {
                        'dates': dates,
                        'title': '',
                        'company': '',
                        'description': []
                    }
                    
                    # Try to extract title and company from the same line
                    clean_line = re.sub(r'\b\w+\s+\d{4}\b', '', line).strip()
                    if clean_line:
                        parts = clean_line.split(' - ')
                        if len(parts) >= 2:
                            current_experience['title'] = parts[0].strip()
                            current_experience['company'] = parts[1].strip()
                        else:
                            current_experience['title'] = clean_line
                
                elif current_experience and line:
                    # Add to description if we have a current experience
                    current_experience['description'].append(line)
        
        # Add the last experience if any
        if current_experience:
            experiences.append(current_experience)
        
        return experiences
    
    def _extract_dates(self, text: str) -> Optional[str]:
        """Extract date ranges from text."""
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            if matches:
                return ' - '.join(matches)
        return None
    
    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume text."""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            re.compile(r'\b(Bachelor|Master|PhD|MBA|MS|BS|BA|MA|Ph\.?D\.?)\b', re.IGNORECASE),
            re.compile(r'\b(Associate|Diploma|Certificate)\b', re.IGNORECASE)
        ]
        
        # University/College patterns
        institution_patterns = [
            re.compile(r'\b(University|College|Institute|School)\b', re.IGNORECASE)
        ]
        
        lines = text.split('\n')
        in_education_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if we're entering education section
            if self.section_patterns['education'].search(line):
                in_education_section = True
                continue
            
            # Check if we're leaving education section
            if in_education_section and any(pattern.search(line) for pattern in self.section_patterns.values() if pattern != self.section_patterns['education']):
                in_education_section = False
                continue
            
            if in_education_section:
                # Check for degree
                degree_match = None
                for pattern in degree_patterns:
                    match = pattern.search(line)
                    if match:
                        degree_match = match.group()
                        break
                
                # Check for institution
                institution_match = None
                for pattern in institution_patterns:
                    match = pattern.search(line)
                    if match:
                        # Extract the full institution name
                        words = line.split()
                        for i, word in enumerate(words):
                            if pattern.search(word):
                                # Take surrounding words to form institution name
                                start = max(0, i-2)
                                end = min(len(words), i+3)
                                institution_match = ' '.join(words[start:end])
                                break
                        break
                
                if degree_match or institution_match:
                    education.append({
                        'degree': degree_match or '',
                        'institution': institution_match or '',
                        'year': self._extract_year(line),
                        'raw_text': line
                    })
        
        return education
    
    def _extract_year(self, text: str) -> Optional[str]:
        """Extract graduation year from text."""
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        match = year_pattern.search(text)
        return match.group() if match else None
    
    def parse_resume(self, file_path: str) -> Dict:
        """Parse complete resume and return structured data."""
        try:
            # Extract text
            text = self.extract_text_from_pdf(file_path)
            if not text:
                return {"error": "Could not extract text from PDF"}
            
            # Extract all information
            parsed_data = {
                'contact_info': self.extract_contact_info(text),
                'skills': self.extract_skills(text),
                'experience': self.extract_experience(text),
                'education': self.extract_education(text),
                'raw_text': text,
                'parsed_at': datetime.now().isoformat()
            }
            
            return parsed_data
            
        except Exception as e:
            return {"error": f"Error parsing resume: {str(e)}"}
    
    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize extracted skills."""
        categorized = {category: [] for category in self.skill_ontology.keys()}
        
        for skill in skills:
            skill_lower = skill.lower()
            for category, skill_list in self.skill_ontology.items():
                if any(skill_lower == s.lower() for s in skill_list):
                    categorized[category].append(skill)
                    break
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}


# Example usage and testing
if __name__ == "__main__":
    parser = ResumeParser()
    
    # Test with a sample resume (you would replace this with actual file path)
    # result = parser.parse_resume("sample_resume.pdf")
    # print(json.dumps(result, indent=2))
    
    # Test skill extraction
    sample_text = """
    I have experience with Python, JavaScript, React, and AWS.
    I worked as a Software Engineer at Google using machine learning and Docker.
    Skills: SQL, MongoDB, Git, Agile, Leadership
    """
    
    skills = parser.extract_skills(sample_text)
    print("Extracted Skills:", skills)
    
    categorized = parser.get_skill_categories(skills)
    print("Categorized Skills:", categorized)
