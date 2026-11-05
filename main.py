
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from ingest import search_research_papers, save_papers


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

REPORT_DIR = BASE_DIR / "generated_reports"
REPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Research Nexus AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM UI DESIGN - PROFESSIONAL LIGHT THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #F5F7F3;
    --panel: #FFFFFF;
    --panel-2: #EEF3EC;
    --line: #E1E8DF;
    --muted: #78877D;
    --text: #26352D;
    --green: #397653;
    --green-light: #E5F0E7;
    --green-dark: #28563B;
}

.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(177, 205, 174, 0.20), transparent 30%),
        linear-gradient(135deg, #F8FAF6 0%, #F1F5EF 100%);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4,
.brand, .hero h1, .section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.035em;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown strong {
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EDF3EB 0%, #E6EEE5 100%);
    border-right: 1px solid #DCE6DA;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #26352D !important;
    letter-spacing: -0.06em;
}

.brand span {
    color: var(--green) !important;
}

.tagline {
    color: #78877D;
    font-size: 10px;
    letter-spacing: 2px;
    margin-top: 4px;
    font-weight: 700;
}

[data-testid="stSidebar"] h3 {
    color: #34473A !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 11px;
    padding: 8px 10px;
    margin: 3px 0;
    color: #46594A !important;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #DCEADD;
    color: #28563B !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #DCEBDD;
    border: 1px solid #BFD6C1;
    color: #28563B !important;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        115deg,
        #FFFFFF 0%,
        #F0F6EF 60%,
        #E7F0E6 100%
    );
    border: 1px solid #DDE8DC;
    border-radius: 24px;
    padding: 38px 40px;
    margin: 4px 0 28px;
    box-shadow: 0 12px 35px rgba(42, 72, 49, 0.07);
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -65px;
    top: -150px;
    border: 1px solid rgba(57, 118, 83, 0.16);
    border-radius: 50%;
    box-shadow:
        0 0 0 35px rgba(57, 118, 83, 0.035),
        0 0 0 75px rgba(57, 118, 83, 0.025);
}

.hero-label {
    color: var(--green);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.5px;
}

.hero h1 {
    color: #26352D !important;
    font-size: clamp(30px, 4vw, 46px);
    line-height: 1.12;
    margin: 13px 0 10px;
}

.hero p {
    color: #68786C !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 760px;
    margin-bottom: 0;
}

.section-title {
    color: #26352D !important;
    font-size: 23px;
    font-weight: 700;
    margin: 16px 0 18px;
}

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 126px;
    padding: 22px;
    background: #FFFFFF;
    border: 1px solid #E1E8DF;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(42, 72, 49, 0.045);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 25px rgba(42, 72, 49, 0.09);
    border-color: #B8D0BA;
}

.metric-label {
    color: #78877D;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.4px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 700;
    color: var(--green);
    margin-top: 12px;
}

.agent-card {
    background: #FFFFFF;
    border: 1px solid #E1E8DF;
    border-radius: 17px;
    padding: 19px 16px;
    min-height: 172px;
    margin-bottom: 10px;
    box-shadow: 0 5px 16px rgba(42, 72, 49, 0.045);
    transition: all 0.2s ease;
}

.agent-card:hover {
    transform: translateY(-4px);
    border-color: #AFCBB3;
    box-shadow: 0 12px 26px rgba(42, 72, 49, 0.09);
}

.agent-number {
    color: var(--green);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.agent-name {
    color: #26352D;
    font-size: 15px;
    font-weight: 700;
    margin-top: 9px;
    line-height: 1.4;
}

.agent-description {
    color: #78877D;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 8px;
}

.paper-card {
    background: #FFFFFF;
    border: 1px solid #E1E8DF;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(42, 72, 49, 0.04);
}

.paper-title {
    color: #26352D;
    font-size: 16px;
    font-weight: 700;
}

.paper-meta {
    color: #78877D;
    font-size: 12px;
    margin-top: 8px;
}

div.stButton > button,
div.stFormSubmitButton > button,
div.stDownloadButton > button {
    background: linear-gradient(110deg, #397653 0%, #4D8A60 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    border: 1px solid #397653;
    border-radius: 11px;
    min-height: 45px;
    box-shadow: 0 5px 14px rgba(57, 118, 83, 0.15);
    transition: all 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
div.stDownloadButton > button:hover {
    background: #28563B;
    color: #FFFFFF !important;
    border-color: #28563B;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(57, 118, 83, 0.20);
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #26352D !important;
    border: 1px solid #D6E0D5 !important;
    border-radius: 10px;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9AA69C !important;
}

.stTextInput label,
.stNumberInput label,
.stSlider label,
.stSelectbox label,
.stTextArea label,
.stMultiSelect label {
    color: #46594A !important;
    font-weight: 600;
}

.stSlider [data-baseweb="slider"] {
    color: var(--green);
}

div[data-testid="stExpander"],
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E1E8DF;
    border-radius: 15px;
}

[data-testid="stExpander"] summary {
    color: #34473A !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #EAF0E8;
    border: 1px solid #DDE6DA;
    border-radius: 12px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: #647467 !important;
    border-radius: 9px;
}

.stTabs [aria-selected="true"] {
    color: #28563B !important;
    background: #FFFFFF;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

a {
    color: #397653 !important;
    font-weight: 600;
}

a:hover {
    color: #28563B !important;
}

hr {
    border-color: #E1E8DF;
}

small,
.stCaption {
    color: #78877D !important;
}

footer {
    visibility: hidden;
}

@media (max-width: 800px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 27px 23px;
    }

    .hero h1 {
        font-size: 30px;
    }
}

/* Premium matte-black and gold palette */
:root {
    color-scheme: dark;
    --bg: #0B0B0D;
    --panel: #151517;
    --panel-2: #1D1D20;
    --line: #33302A;
    --muted: #AAA49A;
    --text: #F3EFE6;
    --green: #D4AF37;
    --green-light: #292417;
    --green-dark: #F0D77A;
}

.stApp {
    background:
        radial-gradient(circle at 88% 0%, rgba(212, 175, 55, 0.10), transparent 30%),
        linear-gradient(135deg, #0B0B0D 0%, #111113 55%, #0C0C0E 100%);
    color: #F3EFE6;
}

h1, h2, h3, h4,
.brand, .hero h1, .section-title,
.stMarkdown p, .stMarkdown li, .stMarkdown strong {
    color: #F3EFE6 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151517 0%, #0D0D0F 100%);
    border-right: 1px solid #33302A;
}

[data-testid="stSidebar"] h3 {
    color: #E8D18A !important;
}

.brand {
    color: #F3EFE6 !important;
}

.brand span,
.tagline {
    color: #D4AF37 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #CEC8BB !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    color: #F5E7B2 !important;
    background: rgba(212, 175, 55, 0.10);
    border-color: rgba(212, 175, 55, 0.35);
}

.hero {
    background: linear-gradient(115deg, #171719 0%, #1C1A16 62%, #242016 100%);
    border-color: rgba(212, 175, 55, 0.30);
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.30);
}

.hero:after {
    border-color: rgba(212, 175, 55, 0.22);
    box-shadow:
        0 0 0 35px rgba(212, 175, 55, 0.045),
        0 0 0 75px rgba(212, 175, 55, 0.025);
}

.hero-label,
.metric-value,
.agent-number {
    color: #D4AF37 !important;
}

.hero h1,
.section-title,
.agent-name,
.paper-title {
    color: #F3EFE6 !important;
}

.hero p,
.agent-description,
.paper-meta,
.metric-label {
    color: #AAA49A !important;
}

.metric-card,
.agent-card,
.paper-card {
    background: linear-gradient(145deg, #19191B, #121214);
    border-color: #33302A;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.20);
}

.metric-card:hover,
.agent-card:hover {
    border-color: rgba(212, 175, 55, 0.55);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.32);
}

div.stButton > button,
div.stFormSubmitButton > button,
div.stDownloadButton > button {
    background: linear-gradient(110deg, #B9942D 0%, #E2C45E 100%);
    color: #17140B !important;
    border-color: #C6A43B;
    box-shadow: 0 6px 18px rgba(212, 175, 55, 0.16);
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
div.stDownloadButton > button:hover {
    background: #E7CB70;
    color: #11100B !important;
    border-color: #E7CB70;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #171719 !important;
    color: #F3EFE6 !important;
    border-color: #464035 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #8F897E !important;
}

.stTextInput label,
.stNumberInput label,
.stSlider label,
.stSelectbox label,
.stTextArea label,
.stMultiSelect label {
    color: #D2CCBF !important;
}

div[data-testid="stExpander"],
div[data-testid="stForm"] {
    background: #141416;
    border-color: #33302A;
}

[data-testid="stExpander"] summary {
    color: #E8D18A !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #171719;
    border-color: #33302A;
}

.stTabs [data-baseweb="tab"] {
    color: #B8B1A4 !important;
}

.stTabs [aria-selected="true"] {
    color: #E7CB70 !important;
    background: #292417;
}

[data-testid="stAlert"] {
    background: #171719;
    border-color: #39352C;
}

a {
    color: #E0C15A !important;
}

a:hover {
    color: #F2DB88 !important;
}

hr {
    border-color: #33302A;
}

small,
.stCaption {
    color: #AAA49A !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

defaults = {
    "papers": [],
    "report": "",
    "topic": "",
    "workflow_completed": False,
    "workflow_error": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# 4. AGENT INFORMATION
# ==========================================================

AGENTS = [
    {
        "number": "01",
        "name": "Discovery Agent",
        "description": "Discovers and organizes authentic academic research papers.",
        "icon": "🔎"
    },
    {
        "number": "02",
        "name": "Summarization Agent",
        "description": "Analyzes research objectives, methods and findings.",
        "icon": "📚"
    },
    {
        "number": "03",
        "name": "Literature Review Agent",
        "description": "Compares research themes and methodologies.",
        "icon": "📊"
    },
    {
        "number": "04",
        "name": "Research Gap Agent",
        "description": "Identifies limitations and future research opportunities.",
        "icon": "💡"
    },
    {
        "number": "05",
        "name": "Report Agent",
        "description": "Creates a structured academic research report.",
        "icon": "📄"
    }
]


def show_agents():

    columns = st.columns(5)

    for column, agent in zip(columns, AGENTS):

        with column:

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-number">
                        AGENT {agent['number']}
                    </div>
                    <div style="font-size:25px;margin-top:8px">
                        {agent['icon']}
                    </div>
                    <div class="agent-name">
                        {agent['name']}
                    </div>
                    <div class="agent-description">
                        {agent['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# 5. CREWAI RESEARCH WORKFLOW
# ==========================================================

def run_research_crew(topic, papers):

    # Use CrewAI's native OpenAI-compatible client for Groq. This strips
    # CrewAI-internal message metadata that Groq does not accept.

    llm = LLM(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://api.groq.com/openai/v1",
        provider="openai",
        temperature=0.3,
        max_tokens=1200
    )

    # Keep prompts compact so the five sequential calls fit Groq's TPM limit.

    paper_information = []
    for index, paper in enumerate(papers, start=1):
        abstract = paper["abstract"]
        if len(abstract) > 500:
            abstract = abstract[:500].rsplit(" ", 1)[0] + "..."
        paper_information.append(
            f"{index}. {paper['title']} ({paper['year']}); "
            f"{', '.join(paper['authors'][:3])}; "
            f"DOI: {paper['doi'] or 'Not available'}; "
            f"Venue: {paper['venue']}; Source: {paper['url']}\n"
            f"Abstract: {abstract}"
        )
    paper_information = "\n\n".join(paper_information)

    # ------------------------------------------------------
    # AGENT 1: DISCOVERY
    # ------------------------------------------------------

    discovery_agent = Agent(
        role="Research Paper Discovery Specialist",

        goal="Validate and organize authentic academic research papers.",

        backstory=(
            "You are an academic research discovery specialist. "
            "You work only with supplied scholarly database records. "
            "You never invent titles, authors, DOIs or publication details."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 2: SUMMARIZATION
    # ------------------------------------------------------

    summarization_agent = Agent(
        role="Academic Research Summarization Expert",

        goal="Summarize research objectives, methodologies and findings.",

        backstory=(
            "You are an academic researcher skilled in understanding "
            "research objectives, methodologies, findings and limitations. "
            "You distinguish abstract-based summaries from full-text analysis."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 3: LITERATURE REVIEW
    # ------------------------------------------------------

    literature_agent = Agent(
        role="Literature Review Specialist",

        goal="Compare research themes, methodologies and findings.",

        backstory=(
            "You are an experienced academic literature review specialist. "
            "You identify similarities, differences and research trends "
            "using the supplied academic papers."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 4: RESEARCH GAP
    # ------------------------------------------------------

    gap_agent = Agent(
        role="Research Gap Analysis Expert",

        goal="Identify evidence-supported research gaps and limitations.",

        backstory=(
            "You are a research gap analysis expert. "
            "You identify limitations and future research directions. "
            "You never present unsupported assumptions as established facts."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ------------------------------------------------------
    # AGENT 5: REPORT
    # ------------------------------------------------------

    report_agent = Agent(
        role="Academic Research Report Writer",

        goal="Prepare a structured academic research assistance report.",

        backstory=(
            "You are an academic report writer. "
            "You prepare clear, structured academic reports "
            "with accurate references and source links."
        ),

        llm=llm,
        verbose=True,
        max_iter=1
    )

    # ======================================================
    # TASK 1: DISCOVERY
    # ======================================================

    discovery_task = Task(

        description=f"""
        Review and organize the following research papers.

        Research Topic: {topic}

        Retrieved Records:
        {paper_information}

        Return:
        1. Paper title
        2. Authors
        3. Publication year
        4. DOI
        5. Publication venue
        6. Source URL

        Do not invent or modify references.
        """,

        expected_output=(
            "An organized list of actual research papers "
            "with original metadata and references."
        ),

        agent=discovery_agent
    )

    # ======================================================
    # TASK 2: SUMMARIZATION
    # ======================================================

    summarization_task = Task(

        description=f"""
        Summarize each supplied paper using its title and abstract:

        {paper_information}

        For each, provide at most three concise bullets for objective,
        method/findings, and reported limitations. Mark unavailable details.
        Treat abstracts as abstracts, not full-text evidence.
        """,

        expected_output=(
            "A concise, source-grounded summary for each supplied paper."
        ),

        agent=summarization_agent,

        context=[discovery_task]
    )

    # ======================================================
    # TASK 3: LITERATURE REVIEW
    # ======================================================

    literature_task = Task(

        description="""
        Prepare a comparative literature review.

        Compare:

        1. Research themes
        2. Methodologies
        3. Technologies and datasets, where reported
        4. Findings
        5. Similarities and differences

        Support comparisons using supplied paper references.
        """,

        expected_output="A concise comparative literature review (under 200 words).",

        agent=literature_agent,

        context=[summarization_task]
    )

    # ======================================================
    # TASK 4: RESEARCH GAP
    # ======================================================

    gap_task = Task(

        description="""
        Analyze the literature review and identify research gaps.

        Include:

        1. Reported limitations
        2. Areas needing further investigation
        3. Unresolved research problems
        4. Possible future research directions

        Link proposed gaps to the supplied research papers.

        Clearly distinguish reported evidence from suggestions.
        """,

        expected_output="Up to five evidence-linked gaps and future directions.",

        agent=gap_agent,

        context=[literature_task]
    )

    # ======================================================
    # TASK 5: FINAL REPORT
    # ======================================================

    report_task = Task(

        description=f"""
        Prepare a complete academic research assistance report.

        Research Topic: {topic}

        Write a concise report (no more than about 800 words) with these
        sections:

        1. Research Topic and Introduction
        2. List of Discovered Research Papers
        3. Individual Paper Summaries
        4. Comparative Literature Review
        5. Research Gaps and Supporting Evidence
        6. Future Research Directions
        7. References with DOI and source links

        Use only the supplied research records and task outputs.

        Do not fabricate references, quotations or findings.

        Mention when conclusions are based only on abstracts.
        """,

        expected_output="A concise, structured Markdown research report.",

        agent=report_agent,

        context=[
            summarization_task,
            literature_task,
            gap_task
        ]
    )

    # ======================================================
    # CREATE CREW
    # ======================================================

    research_crew = Crew(

        agents=[
            discovery_agent,
            summarization_agent,
            literature_agent,
            gap_agent,
            report_agent
        ],

        tasks=[
            discovery_task,
            summarization_task,
            literature_task,
            gap_task,
            report_task
        ],

        process=Process.sequential,

        max_rpm=10,

        verbose=True
    )

    return research_crew.kickoff()


# ==========================================================
# 6. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">RESEARCH<span>NEXUS</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">AI RESEARCH INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "✦  Dashboard",
            "▤  Research Library",
            "◈  AI Agent Workspace",
            "↗  Report Studio"
        ],
        label_visibility="collapsed"
    )
    page = {
        "✦  Dashboard": "Dashboard",
        "▤  Research Library": "Research Library",
        "◈  AI Agent Workspace": "AI Agent Workspace",
        "↗  Report Studio": "Report Studio"
    }[page]

    st.divider()

    st.markdown("### AI CONFIGURATION")

    if API_KEY:
        st.success("Groq API Connected")
    else:
        st.error("Groq API Key Missing")

    st.caption(f"Model: {MODEL}")

    st.divider()

    st.markdown("### DATA SOURCE")

    st.markdown("🟢 OpenAlex + Crossref")

    st.caption("Academic research metadata")

    st.divider()

    st.caption("RESEARCH NEXUS AI")
    st.caption("CrewAI Multi-Agent Research Assistant")


# ==========================================================
# 7. APPLICATION HEADER
# ==========================================================

st.html("""
<div class="hero">

    <div class="hero-label">
        ✦ INTELLIGENT ACADEMIC RESEARCH PLATFORM
    </div>

    <h1>Research smarter.<br>Discover what matters.</h1>

    <p>
        Your AI-powered research workspace — discover authentic academic papers,
        understand findings, compare methodologies and uncover evidence-based
        research gaps with five specialized CrewAI agents.
    </p>

</div>
""")


# ==========================================================
# 8. DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Research Command Center</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PAPERS DISCOVERED</div>
                <div class="metric-value">
                    {len(st.session_state.papers)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">SPECIALIZED AI AGENTS</div>
            <div class="metric-value">05</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        report_status = (
            "READY"
            if st.session_state.report
            else "PENDING"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">REPORT STATUS</div>
                <div class="metric-value" style="font-size:23px">
                    {report_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Start New Research</div>',
        unsafe_allow_html=True
    )

    with st.form("research_form"):

        topic = st.text_input(
            "Research Topic",
            placeholder="Example: Generative AI in Healthcare"
        )

        col1, col2 = st.columns(2)

        with col1:

            number = st.slider(
                "Number of Research Papers",
                min_value=1,
                max_value=20,
                value=5
            )

        with col2:

            start_year = st.number_input(
                "Publication Starting Year",
                min_value=1900,
                max_value=datetime.now().year,
                value=2020
            )

        submitted = st.form_submit_button(
            "🚀 Discover & Analyze Research",
            use_container_width=True
        )

    # ======================================================
    # START RESEARCH
    # ======================================================

    if submitted:

        if not topic.strip():

            st.warning("Please enter a research topic.")

        elif not API_KEY:

            st.error("GROQ_API_KEY is missing from your .env file.")

        else:

            st.session_state.topic = topic
            st.session_state.papers = []
            st.session_state.report = ""
            st.session_state.workflow_completed = False
            st.session_state.workflow_error = ""

            try:

                with st.status(
                    "Research workflow running...",
                    expanded=True
                ) as status:

                    # Step 1: OpenAlex

                    st.write("🔎 Discovering academic research papers...")

                    papers = search_research_papers(
                        topic,
                        number,
                        int(start_year)
                    )

                    if not papers:

                        raise RuntimeError(
                            f"No research papers matched '{topic}' from "
                            f"{int(start_year)} onward. Try broader search "
                            "terms or an earlier starting year."
                        )

                    st.session_state.papers = papers

                    save_papers(
                        papers,
                        str(BASE_DIR / "research_papers.txt")
                    )

                    st.write(
                        f"Retrieved {len(papers)} paper records."
                    )

                    # Step 2: CrewAI

                    st.write("🤖 Initializing five CrewAI agents...")

                    st.write("📚 Starting sequential research analysis...")

                    result = run_research_crew(topic, papers)
                    result_text = str(result).strip()

                    if not result_text or result_text == "None":
                        raise RuntimeError(
                            "The AI workflow returned an empty report. "
                            "Check the CrewAI/Groq output above and try again."
                        )

                    # Step 3: Save report

                    report_file = REPORT_DIR / "research_report.md"

                    report_content = (
                        f"# Research Report: {topic}\n\n"
                        f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
                        f"{result_text}"
                    )

                    report_file.write_text(
                        report_content,
                        encoding="utf-8"
                    )

                    st.session_state.report = report_content

                    st.session_state.workflow_completed = True

                    status.update(
                        label="Research analysis completed!",
                        state="complete",
                        expanded=False
                    )

                st.success("Your research report is ready!")

                st.rerun()

            except Exception as error:

                st.session_state.workflow_error = str(error)

                error_text = str(error)
                if "429" in error_text or "rate_limit_exceeded" in error_text:
                    st.error(
                        "Groq's token-per-minute limit was reached during "
                        "analysis. Your papers were retrieved, but the report "
                        "was not completed. Wait about one minute and retry; "
                        "selecting fewer papers can also reduce token use."
                    )
                    st.caption(f"Provider details: {error_text}")
                else:
                    st.error(f"Research workflow failed: {error_text}")

    # Show agent team

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Your AI Research Team</div>',
        unsafe_allow_html=True
    )

    show_agents()

    if st.session_state.workflow_error:

        st.error(st.session_state.workflow_error)


# ==========================================================
# 9. RESEARCH LIBRARY
# ==========================================================

elif page == "Research Library":

    st.markdown(
        '<div class="section-title">Academic Research Library</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.papers:

        st.info(
            "No research papers available. Start a research project first."
        )

    else:

        st.caption(
            f"Topic: {st.session_state.topic} | "
            f"Total Papers: {len(st.session_state.papers)}"
        )

        search_text = st.text_input(
            "Search discovered papers",
            placeholder="Search by title or author..."
        )

        for index, paper in enumerate(
            st.session_state.papers,
            start=1
        ):

            searchable = (
                paper["title"] + " " +
                " ".join(paper["authors"])
            ).lower()

            if search_text.lower() not in searchable:
                continue

            with st.expander(
                f"📄 {index:02d} | {paper['title']}"
            ):

                st.markdown(
                    f"**Authors:** {', '.join(paper['authors'])}"
                )

                st.markdown(
                    f"**Publication Year:** {paper['year']}"
                )

                st.markdown(
                    f"**Publication Venue:** {paper['venue']}"
                )

                st.markdown(
                    f"**DOI:** {paper['doi'] or 'Not available'}"
                )

                st.markdown(
                    f"[Open paper source]({paper['url']})"
                )

                st.markdown("### Abstract")

                st.write(paper["abstract"])


# ==========================================================
# 10. AI AGENT WORKSPACE
# ==========================================================

elif page == "AI Agent Workspace":

    st.markdown(
        '<div class="section-title">Multi-Agent Intelligence Workspace</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Five specialized CrewAI agents collaborate to complete "
        "the academic research workflow."
    )

    show_agents()

    st.markdown("---")

    st.markdown("### Sequential Agent Workflow")

    workflow_steps = [
        "Discovery Agent – Research paper discovery",
        "Summarization Agent – Academic paper summaries",
        "Literature Review Agent – Research comparison",
        "Research Gap Agent – Gap analysis",
        "Report Agent – Final report preparation"
    ]

    for step in workflow_steps:
        st.markdown(f"✓ {step}")

    if st.session_state.workflow_completed:

        st.success("Latest research workflow completed.")

    else:

        st.info(
            "To run the agents and generate a report, select **Dashboard** "
            "in the sidebar, enter a research topic, and click "
            "**Discover & Analyze Research**."
        )


# ==========================================================
# 11. REPORT STUDIO
# ==========================================================

elif page == "Report Studio":

    st.markdown(
        '<div class="section-title">Research Report Studio</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.report:

        st.info(
            "No report generated yet. Select **Dashboard** in the sidebar, "
            "enter a research topic, and click **Discover & Analyze Research**."
        )

    else:

        st.caption(
            f"Research Topic: {st.session_state.topic}"
        )

        tab1, tab2 = st.tabs(
            ["📖 Report Preview", "⬇ Download Report"]
        )

        with tab1:

            st.markdown(st.session_state.report)

        with tab2:

            st.download_button(
                label="Download Research Report",
                data=st.session_state.report,
                file_name="Research_Nexus_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

            metadata_file = BASE_DIR / "research_papers.txt"

            if metadata_file.exists():

                st.download_button(
                    label="Download Research Paper Metadata",
                    data=metadata_file.read_text(encoding="utf-8"),
                    file_name="Research_Papers.txt",
                    mime="text/plain",
                    use_container_width=True
                )


# ==========================================================
# 12. FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#7890A8;
    font-size:12px;
    padding:15px;
">
    RESEARCH NEXUS AI
    <br>
    BUILT WITH CREWAI  ·  GROQ  ·  OPENALEX
    <br><br>
    Academic findings should be validated against original research papers.
</div>
""", unsafe_allow_html=True)
