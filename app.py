import streamlit as st
import os
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import pdf2image
import google.generativeai as genai
from datetime import datetime
import PyPDF2
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set the path for Poppler (Ensure it's correctly installed)
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\Program Files\poppler-24.02.0\Library\bin")

# Function to create placeholder images instead of using "/api/placeholder/"
def create_placeholder_image(width, height, text="Placeholder"):
    """Create a placeholder image with the given dimensions and text."""
    img = Image.new('RGB', (width, height), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    
    # Try to use a common font that should be available
    try:
        font = ImageFont.truetype("arial.ttf", size=int(height/10))
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", size=int(height/10))
        except IOError:
            font = ImageFont.load_default()
    
    # Draw text in the center
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (width/2, height/2)
    position = ((width-text_width)/2, (height-text_height)/2)
    draw.text(position, text, fill=(130, 130, 130), font=font)
    
    return img

# Function to get Gemini AI response
def get_gemini_response(input_text, pdf_content, prompt):
    """Get response from Gemini AI model."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    pdf_text = pdf_content[0]["data"] if pdf_content else ""
    
    try:
        response = model.generate_content([input_text, pdf_text, prompt])
        return response.text if response else "⚠️ No response received."
    except Exception as e:
        return f"⚠️ Error getting AI response: {str(e)}"

# Function to process the uploaded PDF with error handling
def input_pdf_setup(uploaded_file):
    """Process uploaded PDF and convert to images."""
    if uploaded_file is not None:
        try:
            # Read the file bytes first
            file_bytes = uploaded_file.read()
            
            # Verify it's a valid PDF
            if file_bytes[:4] != b'%PDF':
                st.error("⚠️ The uploaded file is not a valid PDF")
                return [], None
                
            # Reset file pointer
            uploaded_file.seek(0)
            
            images = pdf2image.convert_from_bytes(
                file_bytes,
                poppler_path=POPPLER_PATH,
                fmt='jpeg',
                thread_count=4,
                dpi=200  # Higher quality images
            )
            
            if not images:
                st.error("⚠️ Could not convert PDF to images")
                return [], None
                
            first_page = images[0]
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr = img_byte_arr.getvalue()
            
            pdf_parts = [{
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }]
            
            return pdf_parts, images
            
        except Exception as e:
            st.error(f"⚠️ Error processing PDF: {str(e)}")
            return [], None
    else:
        return [], None

# Extract skills from resume text
def extract_skills(text):
    """Extract common technical skills from resume text."""
    # Common technical skills to look for
    skill_keywords = [
        "Python", "Java", "JavaScript", "HTML", "CSS", "React", "Angular", "Vue", 
        "Node.js", "Express", "Flask", "Django", "SQL", "MySQL", "PostgreSQL", 
        "MongoDB", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "CI/CD",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP",
        "Data Analysis", "Data Science", "Big Data", "Hadoop", "Spark", "ETL",
        "Power BI", "Tableau", "Excel", "R", "C++", "C#", ".NET", "Ruby", "PHP",
        "Agile", "Scrum", "Jira", "REST API", "GraphQL", "Microservices", "DevOps",
        "Linux", "Windows", "MacOS", "Networking", "Security", "Testing", "QA"
    ]
    
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    
    return found_skills

# Function to embed YouTube videos
def embed_youtube_video(video_id, height=315):
    """Generate HTML for embedding a YouTube video."""
    return f"""
    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 20px;">
        <iframe src="https://www.youtube.com/embed/{video_id}" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
        </iframe>
    </div>
    """

# Enhanced resume display with more interactive features
def display_uploaded_resume(uploaded_file):
    """Display and analyze an uploaded resume."""
    try:
        # Create tabs for different views
        resume_tabs = st.tabs(["📄 Preview", "📊 Stats", "🔍 Text Analysis"])
        
        with resume_tabs[0]:
            # Create two columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Resume Preview")
                pdf_content, images = input_pdf_setup(uploaded_file)
                
                if images:
                    # Display thumbnail of first page with zoom option
                    st.image(images[0], use_column_width=True, caption="First Page Preview")
                    
                    # Image enhancement controls
                    with st.expander("🛠️ Image Controls"):
                        rotation = st.slider("Rotate Image", 0, 359, 0, 90, key="img_rotate")
                        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1, key="img_bright")
                        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="img_contrast")
                        
                        if rotation != 0 or brightness != 1.0 or contrast != 1.0:
                            try:
                                # Create a copy of the image for enhancement
                                img = images[0].copy()
                                
                                # Apply rotation if needed
                                if rotation != 0:
                                    img = img.rotate(rotation, expand=True)
                                
                                # Apply brightness if needed
                                if brightness != 1.0:
                                    enhancer = ImageEnhance.Brightness(img)
                                    img = enhancer.enhance(brightness)
                                
                                # Apply contrast if needed
                                if contrast != 1.0:
                                    enhancer = ImageEnhance.Contrast(img)
                                    img = enhancer.enhance(contrast)
                                    
                                st.image(img, caption="Enhanced View", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Could not apply image enhancements: {e}")
                
            with col2:
                st.subheader("Resume Details")
                
                # Show file information
                file_details = {
                    "File Name": uploaded_file.name,
                    "File Size": f"{len(uploaded_file.getvalue()) / 1024:.2f} KB",
                    "Upload Time": st.session_state.get('upload_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "Number of Pages": len(images) if images else "Unknown"
                }
                
                # Create a nice looking info card
                st.markdown("""
                <style>
                    .info-card {
                        background-color: #f8f9fa;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-left: 5px solid #4CAF50;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                info_html = "<div class='info-card'>"
                for key, value in file_details.items():
                    info_html += f"<p><strong>{key}:</strong> {value}</p>"
                info_html += "</div>"
                
                st.markdown(info_html, unsafe_allow_html=True)
                
                # Show all pages in an expander
                if images and len(images) > 1:
                    with st.expander("📑 View All Pages"):
                        tabs = st.tabs([f"Page {i+1}" for i in range(len(images))])
                        for i, tab in enumerate(tabs):
                            with tab:
                                st.image(images[i], use_column_width=True)
                                
                                # Add page-specific download button
                                img_byte_arr = io.BytesIO()
                                images[i].save(img_byte_arr, format='JPEG', quality=90)
                                img_byte_arr = img_byte_arr.getvalue()
                                
                                st.download_button(
                                    label=f"⬇️ Download Page {i+1}",
                                    data=img_byte_arr,
                                    file_name=f"resume_page_{i+1}.jpg",
                                    mime="image/jpeg"
                                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download Original PDF",
                        data=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        mime="application/pdf"
                    )
                
                with col2:
                    # Convert to Word option
                    if st.button("📄 Export as .DOCX"):
                        st.info("Feature coming soon! This will convert your PDF to an editable Word document.")
        
        with resume_tabs[1]:
            st.subheader("📊 Resume Statistics")
            
            # Extract text for analysis
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            
            # Basic text stats
            word_count = len(text.split())
            char_count = len(text)
            sentences = len(re.findall(r'[.!?]+', text))
            
            # Extract skills
            skills = extract_skills(text)
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Words", word_count)
            
            with col2:
                st.metric("Characters", char_count)
                
            with col3:
                st.metric("Sentences", sentences)
            
            # Skills visualization
            if skills:
                st.subheader("Skills Detected")
                
                # Create a horizontal bar chart for skills
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = range(len(skills))
                ax.barh(y_pos, [1] * len(skills), align='center', color='skyblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(skills)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Skills')
                ax.set_title('Skills Detected in Resume')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                
                # Word cloud option
                st.write("Skills overview:", ", ".join(skills))
            else:
                st.warning("No common skills detected automatically. Try using the AI analysis features.")
        
        with resume_tabs[2]:
            st.subheader("🔍 Resume Text Analysis")
            
            # Extract and display text
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            
            if text:
                with st.expander("View Extracted Text", expanded=True):
                    st.text_area("Resume Text", text, height=300)
                    
                    # Copy text button
                    st.button("📋 Copy to Clipboard", key="copy_text", help="Copy text to clipboard")
                    
                # Quick text analysis
                with st.expander("Keyword Analysis"):
                    # Count frequency of words
                    word_freq = {}
                    for word in re.findall(r'\b[A-Za-z][A-Za-z0-9]+\b', text):
                        if word.lower() not in ["the", "and", "a", "to", "in", "of", "for", "with"]:
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    # Sort by frequency
                    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display top words
                    if sorted_freq:
                        st.write("Top words in your resume:")
                        cols = st.columns(4)
                        for i, (word, freq) in enumerate(sorted_freq[:12]):
                            cols[i % 4].metric(word, freq)
                    else:
                        st.warning("No meaningful keywords detected.")
                        
                # Contact information extraction
                with st.expander("Contact Information"):
                    # Find email
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, text)
                    
                    # Find phone - fixed the phone pattern extraction
                    phone_pattern = r'\b(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
                    phones = re.findall(phone_pattern, text)
                    phone_numbers = []
                    
                    # Process the phone matches to get complete numbers
                    for phone_match in phones:
                        if isinstance(phone_match, tuple):
                            # Join the groups to form a complete number
                            full_number = ''.join(part for part in phone_match if part)
                            phone_numbers.append(full_number)
                    
                    # Find URLs
                    url_pattern = r'(https?://[^\s]+)|(www\.[^\s]+)|(linkedin\.com/[^\s]+)'
                    urls = re.findall(url_pattern, text)
                    formatted_urls = []
                    
                    # Process URL matches
                    for url_match in urls:
                        if isinstance(url_match, tuple):
                            # Get the first non-empty match
                            full_url = next((part for part in url_match if part), "")
                            if full_url:
                                formatted_urls.append(full_url)
                    
                    # Display found contact info
                    if emails or phone_numbers or formatted_urls:
                        if emails:
                            st.write("📧 Email:", emails[0])
                        if phone_numbers:
                            st.write("📱 Phone:", phone_numbers[0])
                        if formatted_urls:
                            st.write("🔗 Links:", ", ".join(formatted_urls))
                    else:
                        st.warning("No contact information detected.")
                    
                # Resume structure analysis
                with st.expander("Resume Structure"):
                    # Check for common sections
                    sections = {
                        "Education": any(re.search(r'\beducation\b', text, re.IGNORECASE)),
                        "Experience": any(re.search(r'\bexperience\b|\bwork\b', text, re.IGNORECASE)),
                        "Skills": any(re.search(r'\bskills\b|\bcompetencies\b', text, re.IGNORECASE)),
                        "Projects": any(re.search(r'\bprojects\b', text, re.IGNORECASE)),
                        "Certifications": any(re.search(r'\bcertification\b', text, re.IGNORECASE)),
                        "References": any(re.search(r'\breferences\b', text, re.IGNORECASE)),
                    }
                    
                    for section, present in sections.items():
                        st.write(f"{section}: {'✅' if present else '❌'}")
                    
                    # Warnings for missing sections
                    missing = [s for s, p in sections.items() if not p]
                    if missing:
                        st.warning(f"Your resume appears to be missing these standard sections: {', '.join(missing)}")
                
            else:
                st.error("Could not extract text from this PDF. The file may be scanned or image-based.")
                st.info("Try using the AI analysis features which can process image-based PDFs.")
    
    except Exception as e:
        st.error(f"⚠️ Error analyzing resume: {str(e)}")
        st.error("Please try uploading a different PDF file or check if the file is valid.")

# Function to recommend personalized videos based on skills and job
def display_video_recommendations(job_description="", skills=None):
    """Display personalized video recommendations for resume building."""
    st.subheader("🎥 Resume Building Video Resources")
    
    # Default videos (general resume advice)
    default_videos = [
        {"id": "BYUy1yvjHxE", "title": "How to Write a Winning Resume"},
        {"id": "6r1V3Jk3g2s", "title": "ATS-Friendly Resume Tips"},
        {"id": "tBxWg0qjqIc", "title": "Technical Resume Writing Guide"}
    ]
    
    # Skill-specific videos
    skill_videos = {
        "python": [{"id": "wf-BqAjZb8M", "title": "Advanced Python for Data Science"}],
        "javascript": [{"id": "NCwa_xi0Uuc", "title": "JavaScript for Web Development"}],
        "data": [{"id": "UA-pmkbDtMY", "title": "Data Analysis & Visualization Techniques"}],
        "web": [{"id": "7YS6KYQDSHw", "title": "Full Stack Development Roadmap"}],
        "cloud": [{"id": "3hLmDS179YE", "title": "Cloud Certification Guide"}],
        "devops": [{"id": "UbtB4sMaaNM", "title": "DevOps Best Practices"}]
    }
    
    # Determine which videos to show based on job description and skills
    videos_to_show = default_videos.copy()
    
    # If job description contains certain keywords, add relevant videos
    if job_description:
        job_lower = job_description.lower()
        for keyword, vids in skill_videos.items():
            if keyword in job_lower and len(videos_to_show) < 6:  # Limit to 6 videos total
                videos_to_show.extend(vids)
    
    # Create tabs for video categories
    video_tabs = st.tabs(["Essential Resume Tips", "Industry Specific", "Advanced Techniques"])
    
    with video_tabs[0]:
        # Display first set of videos
        for video in videos_to_show[:2]:
            st.subheader(video["title"])
            st.markdown(embed_youtube_video(video["id"]), unsafe_allow_html=True)
    
    with video_tabs[1]:
        # Display second set of videos
        for video in videos_to_show[2:4]:
            if video:
                st.subheader(video["title"])
                st.markdown(embed_youtube_video(video["id"]), unsafe_allow_html=True)
    
    with video_tabs[2]:
        # Display third set of videos
        for video in videos_to_show[4:6]:
            if video:
                st.subheader(video["title"])
                st.markdown(embed_youtube_video(video["id"]), unsafe_allow_html=True)
    
    # Option to see more videos
    with st.expander("View More Videos"):
        additional_videos = [
            {"id": "z3H4W5b-9k4", "title": "How to Highlight Skills Effectively"},
            {"id": "6I5SFUwUw2c", "title": "Resume Formatting Best Practices"}
        ]
        
        for video in additional_videos:
            st.subheader(video["title"])
            st.markdown(embed_youtube_video(video["id"]), unsafe_allow_html=True)

# Create a feature to compare different resume versions
def resume_version_comparison():
    """Track and compare different versions of a resume."""
    st.subheader("📈 Resume Version Tracker")
    
    # Initialize session state if needed
    if 'resume_versions' not in st.session_state:
        st.session_state.resume_versions = []
    
    # Upload a new version
    new_version = st.file_uploader("Upload a new resume version", type=["pdf"], key="version_uploader")
    
    if new_version:
        # Check if already exists
        if any(v['name'] == new_version.name for v in st.session_state.resume_versions):
            st.warning(f"A version with name '{new_version.name}' already exists.")
        else:
            version_name = st.text_input("Version name (optional)", value=f"Version {len(st.session_state.resume_versions) + 1}")
            if st.button("Save Version"):
                # Store the new version
                version_data = {
                    "name": new_version.name,
                    "display_name": version_name,
                    "data": new_version.getvalue(),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.resume_versions.append(version_data)
                st.success(f"Saved '{version_name}'")
    
    # Display all versions
    if st.session_state.resume_versions:
        st.write(f"You have {len(st.session_state.resume_versions)} saved versions:")
        
        for i, version in enumerate(st.session_state.resume_versions):
            with st.expander(f"{version['display_name']} - {version['date']}"):
                st.write(f"Filename: {version['name']}")
                
                # Preview button
                if st.button(f"Preview", key=f"preview_{i}"):
                    pdf_content, images = input_pdf_setup(io.BytesIO(version['data']))
                    if images:
                        st.image(images[0], use_column_width=True)
                
                # Download button
                st.download_button(
                    f"Download",
                    data=version['data'],
                    file_name=version['name'],
                    mime="application/pdf",
                    key=f"download_{i}"
                )
        
        # Compare versions
        if len(st.session_state.resume_versions) >= 2:
            st.subheader("Compare Versions")
            col1, col2 = st.columns(2)
            with col1:
                version1 = st.selectbox("Select first version", 
                                        range(len(st.session_state.resume_versions)),
                                        format_func=lambda i: st.session_state.resume_versions[i]['display_name'])
            with col2:
                version2 = st.selectbox("Select second version", 
                                        range(len(st.session_state.resume_versions)),
                                        format_func=lambda i: st.session_state.resume_versions[i]['display_name'],
                                        index=min(1, len(st.session_state.resume_versions)-1))
            
            if st.button("Compare Selected Versions"):
                # Extract text from both versions
                pdf1 = PyPDF2.PdfReader(io.BytesIO(st.session_state.resume_versions[version1]['data']))
                pdf2 = PyPDF2.PdfReader(io.BytesIO(st.session_state.resume_versions[version2]['data']))
                
                text1 = "\n".join([page.extract_text() for page in pdf1.pages if page.extract_text()])
                text2 = "\n".join([page.extract_text() for page in pdf2.pages if page.extract_text()])
                
                # Simple comparison
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(st.session_state.resume_versions[version1]['display_name'])
                    _, img1 = input_pdf_setup(io.BytesIO(st.session_state.resume_versions[version1]['data']))
                    if img1:
                        st.image(img1[0], use_column_width=True)
                    
                with col2:
                    st.subheader(st.session_state.resume_versions[version2]['display_name'])
                    _, img2 = input_pdf_setup(io.BytesIO(st.session_state.resume_versions[version2]['data']))
                    if img2:
                        st.image(img2[0], use_column_width=True)
                    
                # Show differences
                unique_to_1 = words1 - words2
                unique_to_2 = words2 - words1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Unique in Version 1")
                    st.write(", ".join(list(unique_to_1)[:50]) if unique_to_1 else "No unique words")
                
                with col2:
                    st.subheader("Unique in Version 2")
                    st.write(", ".join(list(unique_to_2)[:50]) if unique_to_2 else "No unique words")
        else:
            st.info("No resume versions saved yet. Upload your first version to get started.")
# Function to display template previews with actual images
def resume_templates():
    """Display and provide resume templates."""
    st.subheader("📝 Resume Templates")
    
    # Create tabs for different template categories
    template_tabs = st.tabs(["Professional", "Creative", "Technical", "Entry-Level"])
    
    # Template images (replace with your actual image URLs)
    template_images = {
        "Professional": [
            "https://marketplace.canva.com/EAFzfwx_Qik/4/0/1131w/canva-blue-simple-professional-cv-resume-T9RPR4DPdiw.jpg",
            "https://d.novoresume.com/images/doc/general-resume-template.png"
        ],
        "Creative": [
            "https://img.freepik.com/free-psd/designer-template-design_23-2151824715.jpg",
            "https://blog.photoadking.com/wp-content/uploads/2023/03/Creative-Graphic-Designer-Resume.jpg"
        ],
        "Technical": [
            "https://cdn.careerfoundry.com/en/wp-content/uploads/2022/09/Beamjobs-Resume-Example-min.webp",
            "https://assets.qwikresume.com/resume-samples/pdf/screenshots/technical-manager-1562829237-pdf.jpg"
        ],
        "Entry-Level": [
            "https://cdn-images.resumelab.com/pages/entry_level_resumelab_1_1.png",
            "https://d25zcttzf44i59.cloudfront.net/entry-level-sales-resume-example.png"
        ]
    }
    
    for i, tab in enumerate(template_tabs):
        template_type = ["Professional", "Creative", "Technical", "Entry-Level"][i]
        with tab:
            st.write(f"Choose from our collection of {template_type.lower()} resume templates")
            
            # Create a grid layout
            col1, col2 = st.columns(2)
            
            # Display template images
            with col1:
                st.image(template_images[template_type][0], 
                        caption=f"{template_type} Template 1",
                        use_container_width=True)
                st.button(f"Use This Template", key=f"template_{template_type}_1")
            
            with col2:
                st.image(template_images[template_type][1], 
                        caption=f"{template_type} Template 2",
                        use_container_width=True)
                st.button(f"Use This Template", key=f"template_{template_type}_2")
            
            # Template features
            st.write("### Features:")
            st.write(f"- Optimized for {template_type.lower()} roles")
            st.write("- ATS-friendly formatting")
            st.write("- Customizable sections")
            st.write("- Professional design")

# Streamlit App Configuration
st.set_page_config(page_title="ATS Resume Expert", layout="wide")

# App header with custom styling
st.markdown("""
<style>
    .main-header {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 {
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #4B5563;
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>ATS Resume Expert</h1>
    <p>Upload your resume and get instant feedback to improve your chances of landing your dream job.</p>
</div>
""", unsafe_allow_html=True)

# Input fields
candidate_name = st.text_input("Candidate Name:")
if candidate_name:
    st.markdown(f"### **👤 Candidate Name: {candidate_name}**")

input_text = st.text_area("📌 Paste the Job Description:", key="input", height=150)
uploaded_file = st.file_uploader("📂 Upload Your Resume (PDF Only)", type=["pdf"])

# Display uploaded resume if available
if uploaded_file is not None:
    try:
        # Store the uploaded file in session state
        if 'uploaded_resume' not in st.session_state:
            st.session_state.uploaded_resume = uploaded_file
            st.session_state.upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Display the resume
        display_uploaded_resume(uploaded_file)
    except Exception as e:
        st.error(f"Error displaying resume: {str(e)}")

# Submit Buttons
col1, col2, col3 = st.columns(3)

with col1:
    submit1 = st.button("📋 Analyze Resume")

with col2:
    submit2 = st.button("📈 Improve My Skills")

with col3:
    submit3 = st.button("🎯 Check Match Percentage")

# Prompts for Gemini AI
input_prompt1 = """
You are an experienced HR with expertise in Data Science, Full Stack Web Development, 
Big Data Engineering, DevOps, and Data Analysis. Your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the given job.
Highlight the strengths and weaknesses of the applicant in relation to the specified job.
Provide specific feedback on how to improve the resume for this particular job.
"""

input_prompt2 = """
You are a Technical HR Manager with expertise in Data Science, Full Stack Web Development, 
Big Data Engineering, DevOps, and Data Analysis. Your role is to scrutinize the resume in light of the job description.
Provide detailed feedback on the candidate's skills, suggest concrete improvements, and highlight missing competencies.
Recommend specific courses or certifications that would make the candidate more competitive for this role.
"""

input_prompt3 = """
You are an ATS scanner with expertise in Data Science, Web Development, Big Data Engineering, DevOps, and Data Analysis.
Analyze the resume against the job description and provide:
1️⃣ The percentage match for the job (start with a number followed by '%').
2️⃣ A bullet-point list of missing skills and keywords.
3️⃣ Specific recommendations to increase the match percentage.
Ensure the response includes a numerical percentage at the start and is formatted with clear sections.
"""

# Handle Button Clicks
if submit1:
    if uploaded_file is not None and input_text:
        pdf_content, _ = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt1)
        st.subheader("📋 Resume Analysis Report")
        st.write(response)
    elif uploaded_file is None:
        st.warning("Please upload your resume.")
    elif not input_text:
        st.warning("Please paste the job description.")

if submit2:
    if uploaded_file is not None and input_text:
        pdf_content, _ = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt2)
        st.subheader("📈 Skills Improvement Suggestions")
        st.write(response)
        
        # Extract skills and show video
        text = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        full_text = "\n".join([page.extract_text() for page in text.pages if page.extract_text()])
        skills = extract_skills(full_text)
        display_video_recommendations(input_text, skills)

    elif uploaded_file is None:
        st.warning("Please upload your resume.")
    elif not input_text:
        st.warning("Please paste the job description.")

if submit3:
    if uploaded_file is not None and input_text:
        pdf_content, _ = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        st.subheader("🎯 Job Match Percentage")
        st.write(response)
    elif uploaded_file is None:
        st.warning("Please upload your resume.")
    elif not input_text:
        st.warning("Please paste the job description.")

# Additional features
st.sidebar.title("✨ More Tools")
selected_tool = st.sidebar.radio("Choose a tool:", 
                                 ["📝 Resume Templates", "📈 Resume Version Tracker"])

if selected_tool == "📝 Resume Templates":
    resume_templates()
elif selected_tool == "📈 Resume Version Tracker":
    resume_version_comparison()


    
