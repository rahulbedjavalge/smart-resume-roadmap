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
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        openai.api_key = self.api_key
    
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
