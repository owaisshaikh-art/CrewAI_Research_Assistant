
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
# 2. CUSTOM UI DESIGN
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 85% 5%, #172B43 0%, transparent 30%),
        #08111F;
    color: #E8F0FA;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #0D1929;
    border-right: 1px solid #26374B;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    color: white;
    letter-spacing: -1px;
}

.brand span {
    color: #55D6D0;
}

.tagline {
    color: #91A6BB;
    font-size: 11px;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(120deg, #122A40, #17263F 60%, #263054);
    border: 1px solid #314D65;
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 25px;
}

.hero-label {
    color: #55D6D0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px;
    color: white;
    margin: 12px 0;
}

.hero p {
    color: #B4C6D8;
    font-size: 15px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #F2F7FC;
    margin: 20px 0;
}

.metric-card {
    background: #112237;
    border: 1px solid #29435A;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    font-size: 12px;
    color: #94ABC1;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 29px;
    font-weight: 700;
    color: #55D6D0;
    margin-top: 7px;
}

.agent-card {
    background: #112237;
    border: 1px solid #29435A;
    border-radius: 15px;
    padding: 17px;
    min-height: 145px;
    margin-bottom: 10px;
}

.agent-number {
    color: #55D6D0;
    font-size: 11px;
    font-weight: 700;
}

.agent-name {
    color: #F5F8FC;
    font-size: 16px;
    font-weight: 700;
    margin-top: 8px;
}

.agent-description {
    color: #A7B9CC;
    font-size: 12px;
    margin-top: 7px;
}

.paper-card {
    background: #112237;
    border: 1px solid #29435A;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.paper-title {
    font-size: 16px;
    color: #F4F8FC;
    font-weight: 700;
}

.paper-meta {
    color: #93A9BD;
    font-size: 12px;
    margin-top: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg, #24BDB6, #398CBE);
    color: #061522;
    font-weight: 800;
    border: none;
    border-radius: 10px;
    min-height: 45px;
}

div.stButton > button:hover {
    background: #75E2D8;
    color: #061522;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #13263B;
    color: white;
    border-color: #34516B;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    background: #112237;
    border: 1px solid #29435A;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #0D1B2D;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #AFC2D5;
}

.stTabs [aria-selected="true"] {
    color: #55D6D0 !important;
}

hr {
    border-color: #29435A;
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
            "Dashboard",
            "Research Library",
            "AI Agent Workspace",
            "Report Studio"
        ],
        label_visibility="collapsed"
    )

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
        INTELLIGENT ACADEMIC RESEARCH PLATFORM
    </div>

    <h1>Turn Research Into Discovery.</h1>

    <p>
        Discover authentic academic papers, analyze research findings,
        compare methodologies and uncover evidence-based research gaps
        using five specialized AI agents.
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
    POWERED BY CREWAI + GROQ + OPENALEX
    <br><br>
    Academic findings should be validated against original research papers.
</div>
""", unsafe_allow_html=True)
