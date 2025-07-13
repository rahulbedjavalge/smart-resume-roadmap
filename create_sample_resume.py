"""
Sample Resume Creator
Creates a sample PDF resume for testing the application.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

def create_sample_resume():
    """Create a sample PDF resume for testing."""
    filename = "sample_resumes/john_doe_resume.pdf"
    
    # Create the sample_resumes directory if it doesn't exist
    os.makedirs("sample_resumes", exist_ok=True)
    
    # Create the document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.HexColor('#2E86AB')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#A23B72')
    )
    
    # Content
    content = []
    
    # Header
    content.append(Paragraph("John Doe", title_style))
    content.append(Paragraph("Software Engineer", styles['Heading3']))
    content.append(Paragraph("Email: john.doe@email.com | Phone: (555) 123-4567", styles['Normal']))
    content.append(Paragraph("LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe", styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Skills
    content.append(Paragraph("Skills", heading_style))
    skills_text = """
    <b>Programming Languages:</b> Python, JavaScript, Java, SQL<br/>
    <b>Frameworks:</b> React, Django, Flask, Node.js<br/>
    <b>Databases:</b> PostgreSQL, MongoDB, Redis<br/>
    <b>Cloud Platforms:</b> AWS, Docker, Kubernetes<br/>
    <b>Tools:</b> Git, Jenkins, JIRA, Agile
    """
    content.append(Paragraph(skills_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Experience
    content.append(Paragraph("Experience", heading_style))
    
    # Job 1
    content.append(Paragraph("<b>Senior Software Engineer</b> - Tech Corp", styles['Normal']))
    content.append(Paragraph("Jan 2020 - Present", styles['Normal']))
    experience_1 = """
    • Developed and maintained web applications using React and Python Django<br/>
    • Implemented CI/CD pipelines using Jenkins and Docker<br/>
    • Collaborated with cross-functional teams using Agile methodologies<br/>
    • Optimized database queries improving performance by 40%
    """
    content.append(Paragraph(experience_1, styles['Normal']))
    content.append(Spacer(1, 10))
    
    # Job 2
    content.append(Paragraph("<b>Software Developer</b> - StartupCo", styles['Normal']))
    content.append(Paragraph("Jun 2018 - Dec 2019", styles['Normal']))
    experience_2 = """
    • Built REST APIs using Flask and integrated with MongoDB<br/>
    • Developed data processing pipelines using Python and SQL<br/>
    • Implemented authentication and authorization systems<br/>
    • Participated in code reviews and mentored junior developers
    """
    content.append(Paragraph(experience_2, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Education
    content.append(Paragraph("Education", heading_style))
    content.append(Paragraph("<b>Bachelor of Science in Computer Science</b>", styles['Normal']))
    content.append(Paragraph("University of Technology - 2018", styles['Normal']))
    content.append(Spacer(1, 10))
    
    # Projects
    content.append(Paragraph("Projects", heading_style))
    projects_text = """
    <b>E-commerce Platform:</b> Built a full-stack e-commerce application using React, Node.js, and PostgreSQL<br/>
    <b>Data Analysis Tool:</b> Created a Python-based data visualization tool using Pandas and Matplotlib<br/>
    <b>Machine Learning Model:</b> Developed a recommendation system using TensorFlow and scikit-learn
    """
    content.append(Paragraph(projects_text, styles['Normal']))
    
    # Build the document
    doc.build(content)
    print(f"Sample resume created: {filename}")
    return filename

if __name__ == "__main__":
    create_sample_resume()
