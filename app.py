"""
Main Streamlit Application
This file creates the user interface for the Smart Resume Analyzer and Career Roadmap Generator.
"""

import streamlit as st
import os
import json
from resume_parser import ResumeParser
from skill_matcher import SkillMatcher
from roadmap_generator import RoadmapGenerator

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    .stButton>button {
        background-color: #4472c4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Classes ---
@st.cache_resource
def load_resources():
    """Load and cache the necessary class instances."""
    parser = ResumeParser()
    matcher = SkillMatcher()
    # Defer RoadmapGenerator initialization until API key is checked
    return parser, matcher

parser, matcher = load_resources()

# --- Session State ---
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'skill_gaps' not in st.session_state:
    st.session_state.skill_gaps = None
if 'roadmap' not in st.session_state:
    st.session_state.roadmap = None

# --- Sidebar ---
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    if uploaded_file:
        # Save uploaded file to a temporary location
        temp_dir = "temp_resumes"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                st.session_state.resume_data = parser.parse_resume(file_path)
                st.success("Resume analyzed successfully!")
    
    st.markdown("---")
    st.info("This app uses AI to analyze your resume and generate a personalized career roadmap.")

# --- Main Content ---
st.title("🚀 Smart Resume Analyzer & Career Roadmap Generator")
st.markdown("Welcome! Upload your resume to get started.")

if st.session_state.resume_data:
    # --- Display Parsed Resume Details ---
    st.header("📄 Your Resume Details")
    
    data = st.session_state.resume_data
    
    # Contact Info
    st.subheader("Contact Information")
    st.json(data.get('contact_info', {}))
    
    # Skills
    st.subheader("Extracted Skills")
    categorized_skills = parser.get_skill_categories(data.get('skills', []))
    for category, skills in categorized_skills.items():
        st.markdown(f"**{category.replace('_', ' ').title()}:**")
        st.write(", ".join(skills))
    
    # Experience
    st.subheader("Work Experience")
    for exp in data.get('experience', []):
        st.markdown(f"**{exp.get('title', 'N/A')}** at **{exp.get('company', 'N/A')}**")
        st.markdown(f"*{exp.get('dates', 'N/A')}*")
        for desc in exp.get('description', []):
            st.markdown(f"- {desc}")
    
    # Education
    st.subheader("Education")
    for edu in data.get('education', []):
        st.markdown(f"**{edu.get('degree', 'N/A')}** from **{edu.get('institution', 'N/A')}** ({edu.get('year', 'N/A')})")
    
    st.markdown("---")
    
    # --- Career Roadmap Generation ---
    st.header("🗺️ Generate Your Career Roadmap")
    
    # Target Role Selection
    job_roles = list(matcher.job_descriptions.keys())
    target_role = st.selectbox("Select Your Target Role", job_roles)
    
    if st.button("Identify Skill Gaps"):
        with st.spinner("Identifying skill gaps..."):
            st.session_state.skill_gaps = matcher.identify_skill_gaps(data['skills'], target_role)
            st.success("Skill gaps identified!")
    
    if st.session_state.skill_gaps:
        st.subheader(f"Skill Gaps for {st.session_state.skill_gaps['target_role']}")
        st.warning(f"**Missing Skills:** {', '.join(st.session_state.skill_gaps['skill_gaps'])}")
        
        # OpenAI API Key Input
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        
        if st.button("Generate Roadmap"):
            if not api_key:
                st.error("Please enter your OpenAI API Key to generate the roadmap.")
            else:
                os.environ["OPENAI_API_KEY"] = api_key
                try:
                    generator = RoadmapGenerator()
                    with st.spinner("Generating your personalized roadmap... This may take a moment."):
                        roadmap_result = generator.generate_roadmap(
                            current_skills=st.session_state.skill_gaps['current_skills'],
                            target_role=st.session_state.skill_gaps['target_role'],
                            skill_gaps=st.session_state.skill_gaps['skill_gaps']
                        )
                        if "error" not in roadmap_result:
                            st.session_state.roadmap = roadmap_result['roadmap']
                            st.success("Roadmap generated successfully!")
                        else:
                            st.error(roadmap_result['error'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if st.session_state.roadmap:
        st.subheader("Your Personalized 6-Month Roadmap")
        st.markdown(st.session_state.roadmap)

else:
    st.info("Please upload a resume to begin the analysis.")
