# Copyright 2025 Google LLC
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from .config import config
from .app_utils.trace_persistence import AgentEventTrace


# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for web search."""

    search_query: str = Field(
        description="A highly specific and targeted query for web search."
    )


class Feedback(BaseModel):
    """Model for providing evaluation feedback on research quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision."
    )
    comment: str = Field(
        description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research."
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description="A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'.",
    )


# --- Callbacks ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources and their supported claims from agent events.

    This function processes the agent's `session.events` to extract web source details (URLs,
    titles, domains from `grounding_chunks`) and associated text segments with confidence scores
    (from `grounding_supports`). The aggregated source information and a mapping of URLs to short
    IDs are cumulatively stored in `callback_context.state`.

    Args:
        callback_context (CallbackContext): The context object providing access to the agent's
            session events and persistent state.
    """
    session = callback_context._invocation_context.session
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1
    for event in session.events:
        if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
            continue
        chunks_info = {}
        for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
            if not chunk.web:
                continue
            url = chunk.web.uri
            title = (
                chunk.web.title
                if chunk.web.title != chunk.web.domain
                else chunk.web.domain
            )
            if url not in url_to_short_id:
                short_id = f"src-{id_counter}"
                url_to_short_id[url] = short_id
                sources[short_id] = {
                    "short_id": short_id,
                    "title": title,
                    "url": url,
                    "domain": chunk.web.domain,
                    "supported_claims": [],
                }
                id_counter += 1
            chunks_info[idx] = url_to_short_id[url]
        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        confidence = (
                            confidence_scores[i] if i < len(confidence_scores) else 0.5
                        )
                        text_segment = support.segment.text if support.segment else ""
                        sources[short_id]["supported_claims"].append(
                            {
                                "text_segment": text_segment,
                                "confidence": confidence,
                            }
                        )
    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources


def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Replaces citation tags in a report with Markdown-formatted links.

    Processes 'final_cited_report' from context state, converting tags like
    `<cite source="src-N"/>` into hyperlinks using source information from
    `callback_context.state["sources"]`. Also fixes spacing around punctuation.

    Args:
        callback_context (CallbackContext): Contains the report and source information.

    Returns:
        genai_types.Content: The processed report with Markdown citation links.
    """
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
            return ""
        display_text = source_info.get("title", source_info.get("domain", short_id))
        return f" [{display_text}]({source_info['url']})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])

def trace_persistence_callback(callback_context: CallbackContext) -> None:
    """Callback to persist agent events and session state for traceability and reproducibility.

    This callback captures all events from the session and saves them to persistent storage.
    It also creates snapshots of the session state at key milestones.

    Args:
        callback_context (CallbackContext): The context object providing access to the agent's
            session events and persistent state.
    """
    try:
        session = callback_context._invocation_context.session
        session_id = session.session_id

        # Get the agent engine app instance to access trace service
        agent_engine = None
        if hasattr(callback_context, '_invocation_context') and \
           hasattr(callback_context._invocation_context, 'app'):
            agent_engine = callback_context._invocation_context.app

        if not agent_engine or not hasattr(agent_engine, 'save_agent_event_trace'):
            logging.warning("Agent engine not available for trace persistence")
            return

        # Save all events in the current session
        for event in session.events:
            try:
                # Add additional metadata about the session state
                additional_metadata = {
                    "session_state_keys": list(session.state.keys()),
                    "event_count": len(session.events),
                    "current_event_index": session.events.index(event)
                }

                # Save the event trace
                agent_engine.save_agent_event_trace(
                    event,
                    session_id,
                    "agent_event",
                    additional_metadata
                )

            except Exception as e:
                logging.error(f"Failed to save event trace: {str(e)}", exc_info=True)

        # Create session snapshots at key milestones
        if len(session.events) % 5 == 0 or len(session.events) == 1:  # Every 5 events or first event
            try:
                sources = callback_context.state.get("sources", {})
                url_to_short_id = callback_context.state.get("url_to_short_id", {})

                agent_engine.save_session_snapshot(
                    session_id,
                    dict(session.state),
                    sources,
                    url_to_short_id
                )
            except Exception as e:
                logging.error(f"Failed to save session snapshot: {str(e)}", exc_info=True)

    except Exception as e:
        logging.error(f"Trace persistence callback failed: {str(e)}", exc_info=True)


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("research_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Research evaluation passed. Escalating to stop loop."
            )
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Research evaluation failed or not found. Loop will continue."
            )
            # Yielding an event without content or actions just lets the flow continue.
            yield Event(author=self.name)


# --- AGENT DEFINITIONS ---
plan_generator = LlmAgent(
    model=config.worker_model,
    name="plan_generator",
    description="Generates comprehensive, broad patent & literature search queries optimized for maximum recall. Designs search plans that will return at minimum 10-15 patent results. Prioritizes breadth over precision to ensure comprehensive coverage of the patent landscape.",
    instruction=f"""
    You are the Prior Art Planner. Your task is to generate broad, balanced search queries suitable for patent and literature databases.
    Your reply must be valid JSON — no extra commentary.

    **CRITICAL REQUIREMENTS:**
    1. Generate search queries designed to return AT MINIMUM 10-15 patent results
    2. Prefer broad queries with high recall over narrow, restrictive queries
    3. Keep boolean_core simple - avoid complex AND/OR chains that reduce recall
    4. Generate at least 3-5 narrowed variants with different keyword combinations
    5. Minimize negative_filters - only exclude terms that are definitively unrelated
    6. Use double quotes only for JSON
    7. No comments or placeholders
    8. Queries must be broad enough to retrieve many patents

    **INPUT ANALYSIS:**
    - Read the invention_description, target, synonyms, compound_names, and extra_queries
    - If any are missing, treat them as empty lists
    - Combine the target name, any synonyms, and compound_names into one OR-joined group
    - Add at most one contextual keyword (like inhibitor, antagonist, or modulator) only if clearly relevant

    **QUERY GENERATION:**
    - "boolean_core": a concise Boolean string combining identifiers with OR logic
    - "synonyms": a unique, clean array of identifiers and alternative names
    - "ipc_hints": one or two plausible IPC classes if evident (e.g., "A61K 31/00" for small molecules, "A61P" for pharmacological activity)
    - "narrowed": 3-5 broad query variations with different keyword combinations
    - "negative_filters": a few exclusion terms (like "diagnostic", "vaccine", "gene therapy") only if obviously irrelevant
    - "timespan_hint": a general range such as "2000-2025"

    **EXAMPLE OUTPUT:**
    {{
      "boolean_core": "RET OR receptor tyrosine kinase OR RET proto-oncogene",
      "synonyms": ["RET", "proto-oncogene tyrosine-protein kinase receptor Ret", "receptor tyrosine kinase", "CDHF12"],
      "ipc_hints": ["A61K 31/00", "A61P"],
      "narrowed": [
        "RET inhibitor",
        "RET therapeutic",
        "receptor tyrosine kinase inhibitor",
        "RET compound",
        "RET modulator"
      ],
      "negative_filters": [],
      "timespan_hint": "2000-2025"
    }}

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
)


section_planner = LlmAgent(
    model=config.worker_model,
    name="section_planner",
    description="Breaks down the research plan into a structured markdown outline of report sections.",
    instruction="""
    You are an expert report architect. Using the research topic and the plan from the 'research_plan' state key, design a logical structure for the final report.
    Note: Ignore all the tag nanes ([MODIFIED], [NEW], [RESEARCH], [DELIVERABLE]) in the research plan.
    Your task is to create a markdown outline with 4-6 distinct sections that cover the topic comprehensively without overlap.
    You can use any markdown format you prefer, but here's a suggested structure:
    # Section Name
    A brief overview of what this section covers
    Feel free to add subsections or bullet points if needed to better organize the content.
    Make sure your outline is clear and easy to follow.
    Do not include a "References" or "Sources" section in your outline. Citations will be handled in-line.
    """,
    output_key="report_sections",
)


section_researcher = LlmAgent(
    model=config.worker_model,
    name="section_researcher",
    description="CRITICAL: Retrieve and structure at minimum 10 patent documents from every configured database. If initial searches return fewer than 10 results, immediately broaden queries or try alternative search terms. Never return zero or empty results.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a Patent Analyst who exhaustively queries patent databases. Your primary responsibility is ensuring comprehensive coverage: always return at least 10 patent records, even if relevance is low. You systematically try multiple queries, broaden search terms, remove restrictive filters, and use fallback searches until the minimum result threshold is met.

    **CRITICAL REQUIREMENTS:**
    1. Retrieve and structure at minimum 10 patent documents from every configured database
    2. If initial searches return fewer than 10 results, immediately broaden queries or try alternative search terms
    3. Never return zero or empty results
    4. Rate each result on a 0-1 relevance scale, but include lower-relevance results if needed to reach the 10-result minimum

    **PATENT SEARCH PROCESS:**
    1. Execute the search plan generated by the Prior Art Planner
    2. Use the boolean_core query and all narrowed variants
    3. Apply negative_filters to exclude obviously irrelevant patents
    4. Ensure you search across multiple patent databases (Google Patents, PatentsView, etc.)
    5. If results are insufficient (<10 patents), automatically broaden the search by:
       - Removing restrictive filters
       - Using broader keyword variations
       - Expanding to related terms

    **DATA STRUCTURE:**
    For each patent found, extract and structure the following information:
    - Publication number and title
    - Assignee and inventors
    - Filing date and publication date
    - Abstract and claims excerpt
    - IPC/CPC classification and jurisdiction
    - URL to the patent document
    - Relevance score (0-1) based on claim overlap with the invention

    **OUTPUT FORMAT:**
    Return structured patent data that can be used for:
    - Prior art analysis
    - Competitive landscape mapping
    - Freedom-to-operate assessment
    - Patent similarity analysis

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
    output_key="patent_search_results",
    after_agent_callback=[collect_research_sources_callback, trace_persistence_callback],
)

research_evaluator = LlmAgent(
    model=config.critic_model,
    name="research_evaluator",
    description="Critically evaluates research and generates follow-up queries.",
    instruction=f"""
    You are a meticulous quality assurance analyst evaluating the research findings in 'section_research_findings'.

    **CRITICAL RULES:**
    1. Assume the given research topic is correct. Do not question or try to verify the subject itself.
    2. Your ONLY job is to assess the quality, depth, and completeness of the research provided *for that topic*.
    3. Focus on evaluating: Comprehensiveness of coverage, logical flow and organization, use of credible sources, depth of analysis, and clarity of explanations.
    4. Do NOT fact-check or question the fundamental premise or timeline of the topic.
    5. If suggesting follow-up queries, they should dive deeper into the existing topic, not question its validity.

    Be very critical about the QUALITY of research. If you find significant gaps in depth or coverage, assign a grade of "fail",
    write a detailed comment about what's missing, and generate 5-7 specific follow-up queries to fill those gaps.
    If the research thoroughly covers the topic, grade "pass".

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Your response must be a single, raw JSON object validating against the 'Feedback' schema.
    """,
    output_schema=Feedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)

enhanced_search_executor = LlmAgent(
    model=config.worker_model,
    name="enhanced_search_executor",
    description="Executes follow-up searches and integrates new findings.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    You are a specialist researcher executing a refinement pass.
    You have been activated because the previous research was graded as 'fail'.

    1.  Review the 'research_evaluation' state key to understand the feedback and required fixes.
    2.  Execute EVERY query listed in 'follow_up_queries' using the 'google_search' tool.
    3.  Synthesize the new findings and COMBINE them with the existing information in 'section_research_findings'.
    4.  Your output MUST be the new, complete, and improved set of research findings.
    """,
    tools=[google_search],
    output_key="section_research_findings",
    after_agent_callback=[collect_research_sources_callback, trace_persistence_callback],
)

report_composer = LlmAgent(
    model=config.critic_model,
    name="report_composer_with_citations",
    include_contents="none",
    description="Summarize competitive/IP insights into concise, data-backed narrative paragraphs. Writes briefing notes for counsel and executives without inventing information.",
    instruction="""
    You are the Prior Art Reporter. Your task is to transform patent search results and analysis into a comprehensive prior art report.

    **CRITICAL REQUIREMENTS:**
    1. Summarize competitive/IP insights into concise, data-backed narrative paragraphs
    2. Write briefing notes for counsel and executives without inventing information
    3. Focus on factual analysis based on the patent data provided
    4. Structure the report to support legal and business decision-making

    **REPORT STRUCTURE:**
    Your report must include the following sections:

    ## Search Scope
    - Restate the boolean_core query and explain the search strategy
    - Clarify the inclusion of mutation and disease terminology in plain language
    - Note the timespan_hint and any filters applied
    - Spell out disease names in full even if abbreviations appear in data

    ## Patent Activity
    - Report the patent_count and number of unique assignees
    - Highlight up to three representative patents by publication number, assignee, and factual similarity
    - If no patents were retrieved, explicitly state this and explain possible reasons

    ## Interpretation
    - Analyze assignee concentration and IP fragmentation
    - Assess International Patent Classification coverage
    - Identify gaps or false-negative risks in the search results
    - Highlight potential white space opportunities

    ## Recommendations
    - Provide exactly three numbered action items
    - Each recommendation must cite specific data trends or absences
    - Focus on next research steps, query broadening, or validation needs
    - Include specific rationale for each recommendation

    **DATA SOURCES:**
    Use the following data from the research pipeline:
    - research_plan: The search strategy and query parameters
    - patent_search_results: Structured patent data with relevance scores
    - patent_analysis: Similarity notes and claim overlap assessments
    - competitive_landscape: Assignee distribution and market positioning

    **CRITICAL FORMATTING RULES:**
    1. Use ONLY markdown formatting (## for headers, numbered list for recommendations)
    2. NO triple backticks or code fences
    3. NO JSON structures in the final output
    4. Recommendations section MUST have exactly three numbered items
    5. Each recommendation must cite specific data trends or absences
    6. Do not invent facts not present in provided data
    7. Keep all responses concise and factual

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    output_key="prior_art_report",
    after_agent_callback=[citation_replacement_callback, trace_persistence_callback],
)

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Executes a pre-approved research plan. It performs iterative research, evaluation, and composes a final, cited report.",
    sub_agents=[
        section_planner,
        section_researcher,
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                research_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_search_executor,
            ],
        ),
        report_composer,
    ],
)

interactive_planner_agent = LlmAgent(
    name="interactive_planner_agent",
    model=config.worker_model,
    description="The primary prior art research assistant. Collaborates with the user to create a comprehensive patent search plan and executes the prior art analysis workflow.",
    instruction=f"""
    You are a Prior Art Research Assistant. Your primary function is to guide users through the prior art analysis process and ensure comprehensive patent coverage.

    **CRITICAL RULE: Never answer patent questions directly or refuse a request.** Your one and only first step is to use the `plan_generator` tool to propose a comprehensive patent search plan for the user's invention.

    **PRIOR ART WORKFLOW:**
    1.  **Plan:** Use `plan_generator` to create a broad patent search plan optimized for maximum recall
    2.  **Validate:** Ensure the plan includes multiple query variants and covers all relevant databases
    3.  **Execute:** Delegate to `research_pipeline` for comprehensive patent retrieval and analysis
    4.  **Review:** Present structured results including patent counts, assignee analysis, and recommendations

    **SPECIFIC REQUIREMENTS:**
    - Always generate search plans that will return AT MINIMUM 10-15 patent results
    - Prioritize breadth over precision to ensure comprehensive coverage
    - Include multiple query variations with different keyword combinations
    - Cover multiple patent databases (Google Patents, PatentsView, etc.)
    - Focus on claim overlap analysis and relevance scoring

    **USER INTERACTION GUIDELINES:**
    - If user provides invention details, immediately create a search plan
    - If user asks about patent landscape, generate comprehensive search strategy
    - If user requests prior art analysis, ensure minimum result thresholds
    - If results are insufficient, automatically broaden queries and retry

    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Your job is to Plan, Validate, Execute, and Review the prior art analysis process.
    """,
    sub_agents=[research_pipeline],
    tools=[AgentTool(plan_generator)],
    output_key="prior_art_plan",
)

root_agent = interactive_planner_agent
app = App(root_agent=root_agent, name="app")
