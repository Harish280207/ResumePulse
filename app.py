import os

# ============================================================
# MEMORY / CPU OPTIMIZATION
# ============================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ResumePulse | AI Career Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================
# IMPORTANT:
# Set your API key as an environment variable instead of
# placing the key directly inside this file.
#
# PowerShell example:
# $env:GOOGLE_API_KEY="YOUR_NEW_API_KEY"
#
# Or use Streamlit secrets.
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "Google Gemini API key is not configured. "
        "Please set the GOOGLE_API_KEY environment variable."
    )
    st.stop()


# ============================================================
# CACHED GEMINI CLIENT
# ============================================================
# Streamlit reruns the script whenever the user interacts
# with the UI. cache_resource prevents recreating the LLM
# object on every rerun.
# ============================================================

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )


llm = get_llm()


# ============================================================
# CUSTOM UI THEME
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg-main: #fcfbfa;
        --surface: #ffffff;
        --surface-card: #f5f1eb;
        --ink: #1c1917;
        --muted: #78716c;
        --accent: #e67e54;
        --accent-hover: #d06b43;
        --border-color: #e7e2dc;
        --shadow-sm: 0 4px 20px rgba(28, 25, 23, 0.04);
        --shadow-md: 0 10px 30px -5px rgba(230, 126, 84, 0.15);
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--ink);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
        font-family: "Georgia", serif;
    }

    h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #1c1917;
    }

    p, .stMarkdown p, [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #f7f4ef;
        border-right: 1px solid var(--border-color);
    }

    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.25rem;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: #44403c !important;
        font-weight: 500;
    }

    /* Custom Brand Card */
    .brand-card {
        padding: 1.25rem 1rem;
        background: #efece6;
        border: 1px solid var(--border-color);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }

    .brand-name {
        color: #1c1917;
        font-size: 1.25rem;
        font-weight: 700;
        font-family: "Georgia", serif;
        letter-spacing: -0.01em;
    }

    .brand-tagline {
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* Feature & Stat Cards */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
        font-family: "Georgia", serif;
    }

    .stat-label {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 0.2rem;
    }

    .feature-card {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent);
    }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.35rem;
        font-family: "Georgia", serif;
    }

    .feature-desc {
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.5;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fbf9f6;
        border: 2px dashed #d6cec5;
        border-radius: 10px;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: #fdf5f0;
    }

    /* Text Inputs */
    [data-testid="stTextArea"] textarea,
    [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: var(--border-color) !important;
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(230, 126, 84, 0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #e67e54 0%, #d06b43 100%);
        color: #fff7ed;
        font-weight: 650;
        letter-spacing: 0.01em;
        border: 1px solid rgba(255, 247, 237, 0.2);
        box-shadow: 0 4px 16px rgba(230, 126, 84, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #d06b43 0%, #b85a36 100%);
        color: #ffffff;
        box-shadow: 0 6px 22px rgba(230, 126, 84, 0.35);
        transform: translateY(-1px);
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border: 1px solid var(--border-color);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        padding: 0.5rem;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border-color: var(--border-color) !important;
        background-color: var(--surface) !important;
        color: var(--ink) !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        background: var(--surface-card);
        color: var(--ink);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div class="brand-card">
        <div class="brand-name">⚡ ResumePulse</div>
        <div class="brand-tagline">
            Your all-in-one AI career assistant.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

app_mode = st.sidebar.selectbox(
    "Choose a Tool",
    [
        "Resume Checker",
        "Cover Letter Generator",
        "Resume Scorer (JD Matcher)",
        "AI Career Coach",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 Quick Pro-Tips")

st.sidebar.markdown(
    """
    * **PDF Format:** Ensure your resume is exported as a clean text-based PDF.
    * **Tailor JDs:** Paste exact job requirements for precise matching scores.
    * **Iterate:** Use the AI Career Coach to simulate interview questions!
    """
)


# ============================================================
# PDF EXTRACTION
# ============================================================
@st.cache_data(show_spinner=False)
def extract_resume_text(pdf_bytes):
    """
    Extract resume text from PDF.

    The result is cached based on the PDF bytes, so the same
    resume isn't processed repeatedly during Streamlit reruns.
    """

    temp_path = "temp_resume_cached.pdf"

    try:
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        context = " ".join(
            doc.page_content for doc in docs
            if doc.page_content
        )

        return context

    finally:
        # Clean up the temporary PDF.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def load_resume_pdf(uploaded_file):
    """
    Wrapper around the cached PDF extraction function.
    """

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        return ""

    return extract_resume_text(pdf_bytes)


# ============================================================
# HELPER: GET GEMINI RESPONSE TEXT
# ============================================================
def get_response_text(response):
    """
    Safely extract text from LangChain Gemini response.
    """

    if hasattr(response, "text"):
        return response.text

    return str(response.content)


# ============================================================
# 1. RESUME CHECKER
# ============================================================
if app_mode == "Resume Checker":

    st.title("🩺 Resume Health Checker")

    st.markdown(
        "Upload your resume to get a general health score, "
        "strengths, weaknesses, and improvement tips."
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">98%</div>'
            '<div class="stat-label">ATS Parsing Rate</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">6-Point</div>'
            '<div class="stat-label">Comprehensive Audit</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Instant</div>'
            '<div class="stat-label">AI Feedback</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"],
        key="checker_pdf"
    )

    if not uploaded_file:

        st.markdown("### 🌟 What ResumePulse Evaluates For You")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">📈 Health Scoring</div>'
                '<div class="feature-desc">'
                'Get an overall score out of 100 with recruiter verdicts.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f2:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">🔍 Strengths & Gaps</div>'
                '<div class="feature-desc">'
                'Identify standout skills and missing areas to improve.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f3:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">🚀 Career Strategy</div>'
                '<div class="feature-desc">'
                'Receive tailored next steps for your professional growth.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and st.button("Evaluate Resume"):

        with st.spinner("Analyzing resume..."):

            resume_text = load_resume_pdf(uploaded_file)

            prompt = f"""
You are an expert recruiter and resume reviewer.

Analyze the following resume content and provide:

1. A general health score out of 100 with a short verdict.
2. Strong areas (strengths) formatted with clear bullet points.
3. Weak areas (weaknesses) formatted with clear bullet points.
4. Skills explicitly mentioned.
5. Recommended additional skills to add.
6. Suggested next career steps.

Use Markdown headings and formatting so it looks clean and organized.

Resume Content:
{resume_text}
"""

            response = llm.invoke(prompt)

            st.markdown("### 📊 Evaluation Result")

            output_text = get_response_text(response)

            st.markdown(output_text)


# ============================================================
# 2. COVER LETTER GENERATOR
# ============================================================
elif app_mode == "Cover Letter Generator":

    st.title("✍️ AI Cover Letter Generator")

    st.markdown(
        "Generate a tailored professional cover letter based on "
        "your resume and a target job description."
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">300-400</div>'
            '<div class="stat-label">Optimal Word Count</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">100%</div>'
            '<div class="stat-label">Tailored to JD</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Professional</div>'
            '<div class="stat-label">Recruiter Tone</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"],
        key="cover_pdf"
    )

    job_description = st.text_area(
        "Paste Job Description (JD) Here:"
    )

    if not uploaded_file or not job_description:

        st.markdown("### 📝 How Cover Letter Generator Works")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">🔗 Context Matching</div>'
                '<div class="feature-desc">'
                'Combines your resume background with target role '
                'requirements seamlessly.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f2:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">✨ Polished Pitch</div>'
                '<div class="feature-desc">'
                'Writes an engaging opening hook and professional '
                'closing statement.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and job_description and st.button(
        "Generate Cover Letter"
    ):

        with st.spinner("Crafting cover letter..."):

            resume_text = load_resume_pdf(uploaded_file)

            prompt = f"""
Write a professional, compelling cover letter (300-400 words)
tailored specifically to the job description below, using the
candidate's resume context.

Job Description:
{job_description}

Candidate Resume:
{resume_text}
"""

            response = llm.invoke(prompt)

            st.markdown("### ✨ Generated Cover Letter")

            output_text = get_response_text(response)

            st.markdown(output_text)


# ============================================================
# 3. RESUME SCORER + AUTO-UPGRADER
# ============================================================
elif app_mode == "Resume Scorer (JD Matcher)":

    st.title("🎯 Resume & JD Matcher + Auto-Upgrader")

    st.markdown(
        "Compare your resume against a job description, analyze "
        "skill gaps, and instantly generate an upgraded, "
        "ATS-optimized version of your resume."
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Match %</div>'
            '<div class="stat-label">Keyword Precision</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">ATS Score</div>'
            '<div class="stat-label">Compatibility Check</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Instant Upgrade</div>'
            '<div class="stat-label">Auto-Rewrite & Download</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"],
        key="scorer_pdf"
    )

    job_description = st.text_area(
        "Paste Target Job Description Here:",
        key="scorer_jd"
    )

    if not uploaded_file or not job_description:

        st.markdown("### 📌 Maximize Your Interview Callbacks")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">🔑 Keyword Match</div>'
                '<div class="feature-desc">'
                'Find exact words required by ATS scanners.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f2:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">🚀 Auto-Upgrade</div>'
                '<div class="feature-desc">'
                'Instantly rewrite your resume to integrate '
                'missing skills.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f3:
            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">📥 Ready to Download</div>'
                '<div class="feature-desc">'
                'Get your optimized resume file with one click.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and job_description and st.button(
        "Analyze & Upgrade Resume"
    ):

        with st.spinner(
            "Analyzing match and generating your upgraded resume..."
        ):

            resume_text = load_resume_pdf(uploaded_file)

            # ------------------------------------------------
            # 1. MATCH & GAP ANALYSIS
            # ------------------------------------------------
            analysis_prompt = f"""
Analyze how well the candidate's resume matches the target job
description.

Provide:

- **Overall Match Percentage** (e.g., 85%)
- **Score out of 100** based on rubrics
- **Keywords Matched**
- **Missing Keywords**
- **Skill Gap Analysis**

Use Markdown formatting, bold text, and bullet points.

Job Description:
{job_description}

Resume:
{resume_text}
"""

            analysis_response = llm.invoke(analysis_prompt)

            analysis_output = get_response_text(
                analysis_response
            )

            # ------------------------------------------------
            # 2. AUTO-UPGRADE
            # ------------------------------------------------
            upgrade_prompt = f"""
You are an expert resume writer and ATS optimization specialist.

Rewrite and upgrade the candidate's resume to align with the
target job description below.

Requirements:

- Naturally integrate the missing keywords and required competencies.
- Enhance impact statements with strong action verbs.
- Ensure clean, professional structure suitable for professional applications.
- Provide ONLY the polished, upgraded resume text.

Job Description:
{job_description}

Original Resume:
{resume_text}
"""

            upgrade_response = llm.invoke(upgrade_prompt)

            upgraded_output = get_response_text(
                upgrade_response
            )

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------
            st.markdown("### 📊 Match Analysis Report")

            st.markdown(analysis_output)

            st.markdown("---")

            st.markdown("### 🚀 Upgraded & Optimized Resume")

            st.markdown(
                "Your resume has been intelligently rewritten "
                "to bridge all skill gaps and target this "
                "specific job description."
            )

            st.text_area(
                "Upgraded Resume Preview:",
                upgraded_output,
                height=300
            )

            st.download_button(
                label="📥 Download Upgraded Resume (.txt)",
                data=upgraded_output,
                file_name="Upgraded_Resume_ResumePulse.txt",
                mime="text/plain"
            )


# ============================================================
# 4. AI CAREER COACH
# ============================================================
elif app_mode == "AI Career Coach":

    st.title("🧠 AI Career Coach Chatbot")

    st.markdown(
        "Chat with your personal career mentor grounded in "
        "your resume experience."
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">24/7</div>'
            '<div class="stat-label">Mentorship Availability</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Grounded</div>'
            '<div class="stat-label">In Your Real Experience</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-number">Interview</div>'
            '<div class="stat-label">Prep & Strategy</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF) to start coaching",
        type=["pdf"],
        key="coach_pdf"
    )

    if uploaded_file:

        # ------------------------------------------------
        # CACHE PDF EXTRACTION
        # ------------------------------------------------
        resume_context = load_resume_pdf(uploaded_file)

        # Create a stable identifier for the uploaded resume.
        resume_hash = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        # ------------------------------------------------
        # RESET CHAT WHEN A DIFFERENT RESUME IS UPLOADED
        # ------------------------------------------------
        if st.session_state.get("coach_resume_hash") != resume_hash:

            st.session_state.coach_resume_hash = resume_hash

            st.session_state.chat_history = [
                SystemMessage(
                    content=(
                        "You are a professional career counselor "
                        "and resume mentor.\n\n"
                        "Help the candidate with career guidance, "
                        "interview prep, skill gaps, and strategy "
                        "based strictly on their resume context "
                        "below.\n\n"
                        "RESUME CONTEXT:\n\n"
                        f"{resume_context}"
                    )
                )
            ]

        # ------------------------------------------------
        # DISPLAY CHAT HISTORY
        # ------------------------------------------------
        for msg in st.session_state.chat_history[1:]:

            if isinstance(msg, HumanMessage):

                with st.chat_message("user"):
                    st.markdown(msg.content)

            elif isinstance(msg, AIMessage):

                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        # ------------------------------------------------
        # CHAT INPUT
        # ------------------------------------------------
        user_input = st.chat_input(
            "Ask your career coach anything..."
        )

        if user_input:

            with st.chat_message("user"):
                st.markdown(user_input)

            st.session_state.chat_history.append(
                HumanMessage(content=user_input)
            )

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    response = llm.invoke(
                        st.session_state.chat_history
                    )

                    output_text = get_response_text(response)

                    st.markdown(output_text)

                    st.session_state.chat_history.append(
                        AIMessage(content=output_text)
                    )

    else:

        st.info(
            "Please upload your resume PDF in the uploader "
            "above to unlock the AI Career Coach chat session."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 💬 What You Can Ask Your Coach")

        col_f1, col_f2 = st.columns(2)

        with col_f1:

            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">❓ Mock Interviews</div>'
                '<div class="feature-desc">'
                '"Ask me technical interview questions based on '
                'my Python project."'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col_f2:

            st.markdown(
                '<div class="feature-card">'
                '<div class="feature-title">📈 Career Roadmap</div>'
                '<div class="feature-desc">'
                '"What skills should I learn next to transition '
                'into Senior Data Science?"'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
