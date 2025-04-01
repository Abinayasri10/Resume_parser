import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import uuid
import json
import io
import PyPDF2
import matplotlib.pyplot as plt
import base64

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini AI response
def get_gemini_response(input_text, resume_text, prompt):
    """Get response from Gemini AI model."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = model.generate_content([input_text, resume_text, prompt])
        return response.text if response else "⚠️ No response received."
    except Exception as e:
        return f"⚠️ Error getting AI response: {str(e)}"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to create a base64 encoded PDF viewer
def display_pdf(pdf_file):
    """Display the PDF file directly in the Streamlit app."""
    base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display

# Database functions
def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect('resume_database.db')
    cursor = conn.cursor()
    
    # First check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='resumes'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # Create new table with all columns
        cursor.execute('''
        CREATE TABLE resumes (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            role TEXT,
            skills TEXT,
            education TEXT,
            experience TEXT,
            resume_text TEXT,
            upload_date TEXT,
            hr_rating INTEGER DEFAULT 0,
            hr_comments TEXT,
            match_percentage INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Pending'
        )
        ''')
    else:
        # Check if match_percentage column exists, add if not
        cursor.execute("PRAGMA table_info(resumes)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if "match_percentage" not in column_names:
            cursor.execute("ALTER TABLE resumes ADD COLUMN match_percentage INTEGER DEFAULT 0")
        
        # Check if status column exists, add if not
        if "status" not in column_names:
            cursor.execute("ALTER TABLE resumes ADD COLUMN status TEXT DEFAULT 'Pending'")
    
    conn.commit()
    conn.close()

def save_resume_to_db(parsed_data, resume_text, match_percentage=0, status="Pending"):
    """Save parsed resume data to database."""
    conn = sqlite3.connect('resume_database.db')
    cursor = conn.cursor()
    
    # Generate unique ID for the resume
    resume_id = str(uuid.uuid4())
    
    # Extract key fields
    name = parsed_data.get("name", "")
    email = parsed_data.get("email", "")
    phone = parsed_data.get("phone", "")
    role = parsed_data.get("role", "")
    skills = json.dumps(parsed_data.get("skills", []))
    education = json.dumps(parsed_data.get("education", []))
    experience = json.dumps(parsed_data.get("experience", []))
    
    # Insert data
    cursor.execute('''
    INSERT INTO resumes (id, name, email, phone, role, skills, education, 
                        experience, resume_text, upload_date, match_percentage, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (resume_id, name, email, phone, role, skills, education, 
          experience, resume_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
          match_percentage, status))
    
    conn.commit()
    conn.close()
    
    return resume_id

def get_all_resumes():
    """Retrieve all resumes from the database."""
    conn = sqlite3.connect('resume_database.db')
    
    try:
        # First make sure the database has all required columns
        init_db()
        
        df = pd.read_sql_query("SELECT * FROM resumes", conn)
        
        # Convert JSON strings back to lists/dicts
        for col in ['skills', 'education', 'experience']:
            df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else [])
        
        return df
    except Exception as e:
        st.error(f"Error retrieving resumes: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_resume_by_email(email):
    """Retrieve a specific resume by email."""
    conn = sqlite3.connect('resume_database.db')
    
    try:
        # First make sure the database has all required columns
        init_db()
        
        query = "SELECT * FROM resumes WHERE email = ? ORDER BY upload_date DESC LIMIT 1"
        df = pd.read_sql_query(query, conn, params=(email,))
        
        if df.empty:
            return None
        
        # Convert JSON strings back to lists/dicts
        for col in ['skills', 'education', 'experience']:
            df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else [])
        
        return df.iloc[0]
    except Exception as e:
        st.error(f"Error retrieving resume: {str(e)}")
        return None
    finally:
        conn.close()

def update_hr_assessment(resume_id, rating, comments, match_percentage=None, status=None):
    """Update HR assessment for a specific resume."""
    conn = sqlite3.connect('resume_database.db')
    cursor = conn.cursor()
    
    update_fields = []
    params = []
    
    if rating is not None:
        update_fields.append("hr_rating = ?")
        params.append(rating)
    
    if comments is not None:
        update_fields.append("hr_comments = ?")
        params.append(comments)
    
    if match_percentage is not None:
        update_fields.append("match_percentage = ?")
        params.append(match_percentage)
    
    if status is not None:
        update_fields.append("status = ?")
        params.append(status)
    
    if update_fields:
        query = f"UPDATE resumes SET {', '.join(update_fields)} WHERE id = ?"
        params.append(resume_id)
        
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()

# Resume parsing function
def parse_resume(resume_text):
    """Parse resume content using Gemini AI."""
    structured_prompt = """
    Parse the resume text into a structured JSON with the following fields:
    {
        "name": "Candidate's full name",
        "email": "Email address",
        "phone": "Phone number",
        "role": "Current or desired role/position",
        "skills": ["List of skills"],
        "education": [
            {
                "degree": "Degree obtained",
                "institution": "Institution name",
                "year": "Year of completion",
                "gpa": "GPA if available"
            }
        ],
        "experience": [
            {
                "company": "Company name",
                "position": "Position held",
                "duration": "Duration of employment",
                "description": "Brief description of responsibilities"
            }
        ]
    }
    """
    
    response = get_gemini_response("Parse this resume into structured JSON data", 
                                 resume_text, structured_prompt)
    
    try:
        # Find the JSON part in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        parsed_data = json.loads(json_str)
        return parsed_data
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return {}

# Calculate resume match percentage
def calculate_match_percentage(resume_text, job_role):
    """Calculate match percentage against specified job role."""
    prompt = f"""
    Analyze this resume for a {job_role} position and provide:
    1. A match percentage (0-100) for how well this candidate's qualifications match the typical requirements for a {job_role}
    2. Return ONLY the percentage number without any text or explanation
    """
    
    response = get_gemini_response(f"Calculate match percentage for {job_role} position", 
                                  resume_text, prompt)
    
    try:
        # Extract just the number from the response
        import re
        match = re.search(r'(\d+)', response)
        if match:
            percentage = int(match.group(1))
            # Ensure valid percentage range
            return max(0, min(percentage, 100))
        return 0
    except Exception:
        return 0

# Visual representation of skills
def plot_skills(skills_list):
    """Create a visual representation of skills."""
    if not skills_list:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(skills_list))
    
    # Simulate skill levels (would be better with actual data)
    skill_levels = [80 + (i % 4) * 5 for i in range(len(skills_list))]
    
    ax.barh(y_pos, skill_levels, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(skills_list)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Proficiency Level')
    ax.set_title('Skills Proficiency')
    
    plt.tight_layout()
    return fig

# Professional Module (HR Interface)
def professional_module():
    """HR professional interface for viewing and analyzing resumes."""
    st.title("Professional HR Resume Analysis Dashboard")
    
    # Initialize database
    init_db()
    
    # Fetch all resumes
    resumes_df = get_all_resumes()
    
    if resumes_df.empty:
        st.info("No resumes available in the database. Please have candidates upload their resumes first.")
        return
    
    # UI Controls
    st.sidebar.header("HR Controls")
    
    # Filtering options
    filter_option = st.sidebar.selectbox(
        "Filter by:",
        ["All Resumes", "Highest Rated", "Role", "Skills", "Education Level", "Match Percentage"]
    )
    
    filtered_df = resumes_df.copy()
    
    if filter_option == "Highest Rated":
        filtered_df = filtered_df.sort_values(by='hr_rating', ascending=False)
    
    elif filter_option == "Role":
        roles = ["All"] + sorted(filtered_df['role'].unique().tolist())
        selected_role = st.sidebar.selectbox("Select Role:", roles)
        if selected_role != "All":
            filtered_df = filtered_df[filtered_df['role'] == selected_role]
    
    elif filter_option == "Skills":
        # Extract all unique skills
        all_skills = set()
        for skills_list in filtered_df['skills']:
            if isinstance(skills_list, list):
                all_skills.update(skills_list)
        
        selected_skill = st.sidebar.selectbox("Select Skill:", ["All"] + sorted(list(all_skills)))
        
        if selected_skill != "All":
            filtered_df = filtered_df[filtered_df['skills'].apply(
                lambda x: isinstance(x, list) and selected_skill in x
            )]
    
    elif filter_option == "Education Level":
        # Extract degree levels
        degree_levels = ["All", "Bachelor's", "Master's", "PhD", "Associate's", "High School"]
        selected_level = st.sidebar.selectbox("Select Education Level:", degree_levels)
        
        if selected_level != "All":
            filtered_df = filtered_df[filtered_df['education'].apply(
                lambda edu_list: isinstance(edu_list, list) and any(
                    selected_level.lower() in (edu.get('degree', '') or '').lower() 
                    for edu in edu_list if isinstance(edu, dict)
                )
            )]
    
    elif filter_option == "Match Percentage":
        filtered_df = filtered_df.sort_values(by='match_percentage', ascending=False)
    
    # Display resumes in table format
    st.header("Resume Overview")
    
    # Create a simplified table view
    simple_view = pd.DataFrame({
        'Name': filtered_df['name'],
        'Role': filtered_df['role'],
        'Skills': filtered_df['skills'].apply(
            lambda x: ", ".join(x[:3]) + ("..." if len(x) > 3 else "") 
            if isinstance(x, list) else ""
        ),
        'Rating': filtered_df['hr_rating'],
        'Match %': filtered_df['match_percentage'],
        'Status': filtered_df['status']
    })
    
    # Display the table with row selection
    st.dataframe(simple_view, use_container_width=True, height=300)
    
    # Detail view for selected resume
    st.header("Detailed Resume View")
    
    # Select resume by name for detailed view
    selected_name = st.selectbox("Select candidate for detailed view:", 
                                ["Select a candidate"] + filtered_df['name'].tolist())
    
    if selected_name != "Select a candidate":
        # Get the selected resume
        selected_resume = filtered_df[filtered_df['name'] == selected_name].iloc[0]
        
        # Display detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            st.write(f"**Name:** {selected_resume['name']}")
            st.write(f"**Email:** {selected_resume['email']}")
            st.write(f"**Phone:** {selected_resume['phone']}")
            st.write(f"**Role:** {selected_resume['role']}")
            
            st.subheader("Skills")
            skills_list = selected_resume['skills']
            if isinstance(skills_list, list):
                for skill in skills_list:
                    st.write(f"- {skill}")
        
        with col2:
            st.subheader("Education")
            edu_list = selected_resume['education']
            if isinstance(edu_list, list):
                for edu in edu_list:
                    if isinstance(edu, dict):
                        st.write(f"**{edu.get('degree', '')}** - {edu.get('institution', '')}")
                        st.write(f"Year: {edu.get('year', '')}, GPA: {edu.get('gpa', '')}")
            
            st.subheader("Experience")
            exp_list = selected_resume['experience']
            if isinstance(exp_list, list):
                for exp in exp_list:
                    if isinstance(exp, dict):
                        st.write(f"**{exp.get('position', '')}** at {exp.get('company', '')}")
                        st.write(f"Duration: {exp.get('duration', '')}")
                        st.write(f"Description: {exp.get('description', '')}")
        
        # AI Analysis
        st.subheader("AI Resume Analysis")
        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing resume..."):
                analysis = get_gemini_response(
                    "Analyze this resume for an HR professional",
                    selected_resume['resume_text'],
                    """Provide a comprehensive HR analysis of this resume with the following:
                    1. Overall assessment of candidate strengths and weaknesses
                    2. Suitability for the specified role
                    3. Educational qualifications analysis
                    4. Skills assessment
                    5. Experience evaluation
                    6. Suggested interview questions
                    7. Recommended next steps"""
                )
                st.write(analysis)
        
        # HR Assessment Form
        st.subheader("HR Assessment")
        hr_rating = st.slider("Rate this candidate (1-10)", 1, 10, 
                               int(selected_resume['hr_rating']) if selected_resume['hr_rating'] else 5)
        
        match_percentage = st.slider("Match Percentage", 0, 100, 
                                    int(selected_resume.get('match_percentage', 0)))
        
        status_options = ["Pending", "Under Review", "Shortlisted", "Rejected", "Hired"]
        current_status = selected_resume.get('status', 'Pending')
        status = st.selectbox("Application Status", 
                             status_options, 
                             status_options.index(current_status) if current_status in status_options else 0)
        
        hr_comments = st.text_area("HR Comments", 
                                  selected_resume.get('hr_comments', ''))
        
        if st.button("Save Assessment"):
            update_hr_assessment(selected_resume['id'], hr_rating, hr_comments, match_percentage, status)
            st.success("Assessment saved successfully!")
            st.rerun()  # Refresh the page to show updated data

# Resume template suggestion
def suggest_resume_template(role):
    """Suggest resume templates based on the role."""
    templates = {
        "Software Engineer": "https://resumegenius.com/resume-templates/software-engineer-resume-example",
        "Data Scientist": "https://resumegenius.com/resume-templates/data-scientist-resume-example",
        "Marketing": "https://resumegenius.com/resume-templates/marketing-resume-example",
        "Sales": "https://resumegenius.com/resume-templates/sales-resume-example",
        "Finance": "https://resumegenius.com/resume-templates/finance-resume-example",
        "Default": "https://resumegenius.com/resume-templates/professional-resume-template"
    }
    
    # Find the best matching template or return default
    for key in templates.keys():
        if key.lower() in role.lower():
            return templates[key]
    
    return templates["Default"]

# Resume Upload Module (For students)
def resume_upload_module():
    """Module for students to upload and analyze their resumes."""
    st.title("Resume Analyzer")
    
    # Initialize database
    init_db()
    
    # Email for returning users
    email_input = st.text_input("Enter your email address")
    existing_resume = None
    
    if email_input:
        existing_resume = get_resume_by_email(email_input)
        if existing_resume is not None:
            st.success(f"Welcome back, {existing_resume['name']}!")
            
            # Show previous analysis
            st.subheader("Your Previous Resume Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display match percentage with gauge - safely handle missing keys
                match_percentage = existing_resume.get('match_percentage', 0)
                st.metric("Match Percentage", f"{match_percentage}%")
                
                status = existing_resume.get('status', 'Pending')
                st.write(f"**Status:** {status}")
                
                hr_rating = existing_resume.get('hr_rating', 0)
                st.write(f"**HR Rating:** {hr_rating}/10")
                
                hr_comments = existing_resume.get('hr_comments', '')
                if hr_comments:
                    st.write(f"**HR Comments:** {hr_comments}")
            
            with col2:
                # Display skills visualization
                if existing_resume['skills']:
                    skills_chart = plot_skills(existing_resume['skills'])
                    if skills_chart:
                        st.pyplot(skills_chart)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text:
                st.error("Could not extract text from the PDF. Please try another file.")
                return
            
            # Display PDF
            st.subheader("Your Uploaded Resume")
            st.markdown(display_pdf(uploaded_file), unsafe_allow_html=True)
            
            # Show extracted text in an expander
            with st.expander("View Extracted Text"):
                st.text(resume_text)
        
        # Parse resume
        with st.spinner("Analyzing resume..."):
            parsed_data = parse_resume(resume_text)
            
            if not parsed_data:
                st.error("Could not parse resume information. Please try again.")
                return
            
            # Job role selection for match calculation
            st.subheader("Calculate Match Percentage")
            target_role = st.text_input("Enter your target job role/position:", 
                                        value=parsed_data.get('role', ''))
            
            if st.button("Calculate Match"):
                with st.spinner("Calculating match percentage..."):
                    match_percentage = calculate_match_percentage(resume_text, target_role)
                    status = "Pending"
                    
                    # Determine initial status based on match percentage
                    if match_percentage >= 80:
                        status = "Shortlisted"
                    elif match_percentage >= 50:
                        status = "Under Review"
                    else:
                        status = "Needs Improvement"
                    
                    # Save to database with match percentage
                    resume_id = save_resume_to_db(parsed_data, resume_text, match_percentage, status)
                    
                    # Display match information
                    st.success(f"Your resume has a {match_percentage}% match with the {target_role} position!")
                    st.info(f"Status: {status}")
            else:
                # Save to database without match percentage
                resume_id = save_resume_to_db(parsed_data, resume_text)
        
        # Display parsed information
        st.success("Resume uploaded and analyzed successfully!")
        
        # Create tabs for different sections
        info_tab, skills_tab, feedback_tab, resources_tab = st.tabs([
            "Resume Information", "Skills Analysis", "AI Feedback", "Resources"
        ])
        
        with info_tab:
            st.subheader("Extracted Information")
            st.write(f"**Name:** {parsed_data.get('name', '')}")
            st.write(f"**Email:** {parsed_data.get('email', '')}")
            st.write(f"**Phone:** {parsed_data.get('phone', '')}")
            st.write(f"**Role:** {parsed_data.get('role', '')}")
            
            st.subheader("Education")
            for edu in parsed_data.get('education', []):
                st.write(f"**{edu.get('degree', '')}** - {edu.get('institution', '')}")
                st.write(f"Year: {edu.get('year', '')}, GPA: {edu.get('gpa', '')}")
            
            st.subheader("Experience")
            for exp in parsed_data.get('experience', []):
                st.write(f"**{exp.get('position', '')}** at {exp.get('company', '')}")
                st.write(f"Duration: {exp.get('duration', '')}")
                st.write(f"Description: {exp.get('description', '')}")
        
        with skills_tab:
            st.subheader("Skills Analysis")
            
            # Display skills list
            skills_list = parsed_data.get('skills', [])
            if skills_list:
                # Plot skills visualization
                skills_chart = plot_skills(skills_list)
                if skills_chart:
                    st.pyplot(skills_chart)
                
                # Skills summary
                st.subheader("Your Skills")
                for skill in skills_list:
                    st.write(f"- {skill}")
            else:
                st.warning("No skills detected in your resume.")
        
        with feedback_tab:
            st.subheader("AI Resume Feedback")
            if st.button("Get AI Feedback", key="feedback_button"):
                with st.spinner("Analyzing your resume..."):
                    feedback = get_gemini_response(
                        "Provide resume feedback",
                        resume_text,
                        """Give detailed feedback on this resume with:
                        1. Overall assessment (strengths and weaknesses)
                        2. Content improvements (3 specific suggestions)
                        3. Format and presentation tips
                        4. ATS optimization recommendations
                        5. Industry-specific advice"""
                    )
                    st.write(feedback)
        
        with resources_tab:
            st.subheader("Resume Resources")
            
            # Template suggestions
            st.write("### Perfect Resume Templates")
            role = parsed_data.get('role', '')
            template_url = suggest_resume_template(role)
            st.write(f"Based on your role as a {role}, we recommend this template:")
            st.markdown(f"[Professional Resume Template]({template_url})")
            
            # Video recommendations
            st.write("### Video Recommendations")
            st.write("Watch these videos to improve your resume:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.video("https://www.youtube.com/watch?v=y8YH0Qbu5h4")
                st.caption("Resume Tips: How to Create a Resume That Stands Out")
            
            with col2:
                st.video("https://www.youtube.com/watch?v=JOw-dEXCrGM")  
                st.caption("How to Pass the ATS Resume Scan")

# Main application with role selection
def main():
    st.sidebar.title("Resume Analyzer")
    
    # Role selection
    user_role = st.sidebar.radio("Select your role:", ["Student/Candidate", "HR Professional"])
    
    if user_role == "Student/Candidate":
        resume_upload_module()
    else:
        professional_module()

if __name__ == "__main__":
    main()