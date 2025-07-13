"""
Roadmap Generator Module
Generates a personalized 6-month career roadmap using an LLM.
"""

import openai
from typing import Dict, List
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class RoadmapGenerator:
    def __init__(self):
        """Initialize the roadmap generator with OpenAI API key."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and self.api_key != "your_openai_api_key_here":
            openai.api_key = self.api_key
            self.use_openai = True
        else:
            self.use_openai = False
            print("OpenAI API key not configured. Using fallback roadmap generation.")
    
    def generate_roadmap(self, current_skills: List[str], target_role: str, skill_gaps: List[str]) -> str:
        """Generate a career roadmap using OpenAI or fallback method."""
        if self.use_openai:
            return self._generate_openai_roadmap(current_skills, target_role, skill_gaps)
        else:
            return self._generate_fallback_roadmap(current_skills, target_role, skill_gaps)
    
    def _generate_openai_roadmap(self, current_skills: List[str], target_role: str, skill_gaps: List[str]) -> str:
        """Generate roadmap using OpenAI API."""
        prompt = self.generate_roadmap_prompt(current_skills, target_role, skill_gaps)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful career advisor. Generate a structured 6-month roadmap in Markdown format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_roadmap(current_skills, target_role, skill_gaps)
    
    def _generate_fallback_roadmap(self, current_skills: List[str], target_role: str, skill_gaps: List[str]) -> str:
        """Generate a fallback roadmap when OpenAI API is not available."""
        roadmap = f"""
# 🚀 6-Month Career Roadmap for {target_role}

**Current Skills:** {', '.join(current_skills)}  
**Target Role:** {target_role}  
**Key Skill Gaps:** {', '.join(skill_gaps[:5])}  

---

## 📅 Month 1: Foundation Building
**🎯 Focus:** {skill_gaps[0] if skill_gaps else 'Core fundamentals'}

**📚 Learning Resources:**
- Online courses (Coursera, Udemy, edX, YouTube)
- Official documentation and tutorials
- Community forums and Discord servers

**💻 Project Ideas:**
- Build a simple project using the new skill
- Follow along with tutorials and modify them

**🎯 Goals:**
- Understand basic concepts and syntax
- Complete at least 1 hands-on project
- Join relevant online communities

---

## 📅 Month 2: Practical Application
**🎯 Focus:** {skill_gaps[1] if len(skill_gaps) > 1 else 'Hands-on practice'}

**📚 Learning Resources:**
- Practice platforms (LeetCode, HackerRank, Codewars)
- GitHub repositories and open source projects
- Technical blogs and case studies

**💻 Project Ideas:**
- Create a portfolio project from scratch
- Contribute to open source projects
- Build something that solves a real problem

**🎯 Goals:**
- Apply skills in real-world scenarios
- Start building a portfolio
- Network with other developers

---

## 📅 Month 3: Advanced Concepts
**🎯 Focus:** {skill_gaps[2] if len(skill_gaps) > 2 else 'Advanced topics and best practices'}

**📚 Learning Resources:**
- Advanced courses and specializations
- Technical books and research papers
- Webinars and conference talks

**💻 Project Ideas:**
- Build a more complex, full-stack application
- Implement advanced features and optimizations
- Document your learning journey

**🎯 Goals:**
- Master intermediate to advanced concepts
- Understand industry best practices
- Start preparing for technical interviews

---

## 📅 Month 4: Integration & Systems
**🎯 Focus:** System design and technology integration

**📚 Learning Resources:**
- System design courses and books
- Cloud platform documentation (AWS, Azure, GCP)
- DevOps and deployment guides

**💻 Project Ideas:**
- Deploy applications to cloud platforms
- Implement CI/CD pipelines
- Build scalable system architectures

**🎯 Goals:**
- Understand how technologies work together
- Learn deployment and scaling
- Build production-ready applications

---

## 📅 Month 5: Industry Standards & Specialization
**🎯 Focus:** Professional development and specialization

**📚 Learning Resources:**
- Industry certifications and courses
- Professional workshops and bootcamps
- Networking events and meetups

**💻 Project Ideas:**
- Contribute to significant open source projects
- Build applications using industry standards
- Create comprehensive documentation

**🎯 Goals:**
- Meet professional industry standards
- Obtain relevant certifications
- Build a strong professional network

---

## 📅 Month 6: Portfolio & Job Preparation
**🎯 Focus:** Career preparation and job search

**📚 Learning Resources:**
- Interview preparation platforms
- Resume and portfolio optimization guides
- Mock interview sessions

**💻 Project Ideas:**
- Finalize and polish portfolio projects
- Create case studies for your work
- Prepare technical presentation demos

**🎯 Goals:**
- Complete professional portfolio
- Master technical interviews
- Apply for {target_role} positions

---

## 🛠️ Additional Resources

### 📖 **Recommended Books:**
- Technical books specific to your target role
- Software engineering best practices
- Industry-specific knowledge

### 👥 **Communities & Networking:**
- LinkedIn professional groups
- Reddit communities (r/programming, role-specific subreddits)
- Local tech meetups and conferences
- Discord/Slack communities

### 🏆 **Certifications to Consider:**
- Cloud platform certifications (AWS, Azure, GCP)
- Technology-specific certifications
- Project management certifications

### 📊 **Success Metrics:**
- ✅ Complete 2-3 substantial projects per month
- ✅ Contribute to at least 1 open source project
- ✅ Build a portfolio with 5+ quality projects
- ✅ Network with 10+ professionals in your field
- ✅ Practice coding problems regularly (if applicable)
- ✅ Apply to 5+ relevant job positions by month 6

---

## 💡 **Pro Tips:**
1. **Consistency is key** - Dedicate 1-2 hours daily to learning
2. **Build in public** - Share your learning journey on social media
3. **Seek feedback** - Get code reviews and portfolio feedback
4. **Stay updated** - Follow industry trends and new technologies
5. **Practice interviews** - Regular mock interviews with peers

---

*Note: This roadmap is generated using a template. For a more personalized roadmap, consider setting up an OpenAI API key in your .env file.*
"""
        return roadmap
    
    def generate_roadmap_prompt(self, current_skills: List[str], target_role: str, skill_gaps: List[str]) -> str:
        """Create a detailed prompt for the LLM to generate a career roadmap."""
        
        prompt = f"""
        **Objective:** Generate a personalized 6-month career roadmap for a professional aiming to transition into a new role.

        **Candidate Profile:**
        - **Current Skills:** {', '.join(current_skills)}
        - **Desired Job Role:** {target_role}
        - **Identified Skill Gaps:** {', '.join(skill_gaps)}

        **Instructions:**
        Create a structured, month-by-month roadmap that helps the candidate acquire the necessary skills and experience to successfully land a job as a {target_role}. The roadmap should be practical, actionable, and include a mix of learning, project work, and career development activities.

        **Output Format:**
        Please provide the output in a structured Markdown format, with clear headings for each month. For each month, include:
        1.  **Focus Area:** The main theme or skill to focus on.
        2.  **Key Topics:** Specific concepts, tools, or technologies to learn.
        3.  **Project Idea:** A hands-on project to apply the learned skills.
        4.  **Recommended Resources:** Suggest 1-2 high-quality online courses, books, or tutorials.
        5.  **Career Development:** Actionable steps like updating a resume, networking, or preparing for interviews.

        **Example Structure:**

        ### Month 1: Foundational Skills
        -   **Focus Area:** ...
        -   **Key Topics:** ...
        -   **Project Idea:** ...
        -   **Recommended Resources:** ...
        -   **Career Development:** ...

        ---

        ### Month 2: Intermediate Concepts
        -   **Focus Area:** ...
        -   ...

        **Tone:** The tone should be encouraging, professional, and highly practical. Assume the candidate is motivated and can dedicate 10-15 hours per week to this plan.
        """
        return prompt
    
    def generate_roadmap(self, current_skills: List[str], target_role: str, skill_gaps: List[str]) -> Dict:
        """Generate the career roadmap using the OpenAI API."""
        
        prompt = self.generate_roadmap_prompt(current_skills, target_role, skill_gaps)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert career coach and technical mentor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                n=1,
                stop=None,
            )
            
            roadmap_content = response.choices[0].message.content
            
            return {
                "target_role": target_role,
                "roadmap": roadmap_content,
                "metadata": {
                    "model_used": "gpt-3.5-turbo",
                    "prompt_length": len(prompt),
                    "response_length": len(roadmap_content)
                }
            }
            
        except Exception as e:
            print(f"Error generating roadmap: {e}")
            return {"error": "Failed to generate roadmap due to an API error."}

# Example usage and testing
if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY set in a .env file
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with OPENAI_API_KEY=<your_key>")
    else:
        generator = RoadmapGenerator()
        
        # Sample data
        sample_skills = ["Python", "SQL", "Data Analysis", "Communication"]
        sample_target_role = "AI Engineer"
        sample_skill_gaps = ["TensorFlow", "PyTorch", "MLOps", "Model Deployment", "Docker"]
        
        # Generate roadmap
        roadmap_result = generator.generate_roadmap(sample_skills, sample_target_role, sample_skill_gaps)
        
        if "error" not in roadmap_result:
            print(f"--- Roadmap for {sample_target_role} ---")
            print(roadmap_result['roadmap'])
            print("\n--- Metadata ---")
            print(json.dumps(roadmap_result['metadata'], indent=2))
        else:
            print(roadmap_result['error'])
