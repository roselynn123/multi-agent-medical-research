import hashlib
import time
from fastapi import WebSocket
from typing import Any

from gpt_researcher import GPTResearcher

from gpt_researcher.utils.llm import create_chat_completion
import json
class BasicReport:
    def __init__(
        self,
        query: str,
        query_domains: list,
        report_type: str,
        report_source: str,
        source_urls,
        document_urls,
        tone: Any,
        config_path: str,
        websocket: WebSocket,
        headers=None,
        mcp_configs=None,
        mcp_strategy=None,
        max_search_results=None,
    ):
        self.query = query
        self.query_domains = query_domains
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.tone = tone
        self.config_path = config_path
        self.websocket = websocket
        self.headers = headers or {}
        
        # Generate a unique research ID for this report
        self.research_id = self._generate_research_id(query)

        # Initialize researcher with optional MCP parameters
        gpt_researcher_params = {
            "query": self.query,
            "query_domains": self.query_domains,
            "report_type": self.report_type,
            "report_source": self.report_source,
            "source_urls": self.source_urls,
            "document_urls": self.document_urls,
            "tone": self.tone,
            "config_path": self.config_path,
            "websocket": self.websocket,
            "headers": self.headers,
        }

        # Add MCP parameters if provided
        if mcp_configs is not None:
            gpt_researcher_params["mcp_configs"] = mcp_configs
        if mcp_strategy is not None:
            gpt_researcher_params["mcp_strategy"] = mcp_strategy

        self.gpt_researcher = GPTResearcher(**gpt_researcher_params)

        # Override max_search_results_per_query if provided by user
        if max_search_results is not None:
            self.gpt_researcher.cfg.max_search_results_per_query = int(max_search_results)

    def _generate_research_id(self, query: str) -> str:
        """Generate a unique research ID from query and timestamp."""
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"research_{timestamp}_{query_hash}"

    async def _reflect_on_report(self, draft_report: str) -> dict:
        """🔍 Conduct LLM-based reflection and critique on the draft report."""
        if self.websocket:
            await self.websocket.send_json({
                "type": "logs",
                "content": "info",
                "output": "🤔 [Reflection] Draft completed. Agent is reviewing and evaluating report quality..."
            })

        # Retrieve gathered context
        context = " ".join(self.gpt_researcher.context) if hasattr(self.gpt_researcher, 'context') else ""

        reflection_prompt = f"""You are a rigorous peer reviewer and expert in medical literature. Evaluate the following draft report generated from reference materials to determine if it thoroughly, accurately, and comprehensively answers the user's primary medical query.

        User Query: "{self.query}"

        Reference Context Sample:
        {context[:2000]}

        Draft Report to Review:
        {draft_report}

        Identify any logical gaps, missing clinical insights, or key medical aspects that were overlooked.
        Output your response strictly as valid JSON (without any markdown code block markup like ```json ... ```):
        {{
            "is_sufficient": true or false, // Set to true if the report is sufficiently complete and accurate
            "score": integer between 1 and 10, // Quality score
            "critique": "A brief summary of deficiencies or recommendations for improvement",
            "missing_aspects": ["Overlooked clinical detail 1", "Missing medical nuance 2"] // Empty list if is_sufficient is true
        }}
        """
        try:
            # Invoke LLM instance initialized in GPTResearcher
            # response = await self.gpt_researcher.llm.ainvoke(reflection_prompt)
            # response_text = response.content if hasattr(response, 'content') else str(response)
            messages = [{"role": "user", "content": reflection_prompt}]
            response_text = await create_chat_completion(
                messages=messages,
                model=self.gpt_researcher.cfg.smart_llm_model,
                temperature=0.2,
                llm_provider=self.gpt_researcher.cfg.smart_llm_provider,
                cost_callback=self.gpt_researcher.add_costs,
            )
            
            # Sanitize JSON string
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            reflection_data = json.loads(cleaned_text)
            
            print(f"\n📊 [Reflection Score]: {reflection_data.get('score')}/10 | Pass: {reflection_data.get('is_sufficient')}")
            print(f"💬 [Reflection Critique]: {reflection_data.get('critique')}\n")
            
            return reflection_data
        except Exception as e:
            print(f"⚠️ [Reflection Parser Error]: {e}")
            return {"is_sufficient": True, "score": 8, "critique": "Default pass due to parsing failure", "missing_aspects": []}

    async def _refine_report(self, draft_report: str, reflection_review: dict) -> str:
        """🛠️ Refine and enhance the draft report based on reflection critique."""
        critique = reflection_review.get("critique", "")
        missing_aspects = reflection_review.get("missing_aspects", [])

        if self.websocket:
            await self.websocket.send_json({
                "type": "logs",
                "content": "info",
                "output": f"✏️ [Reflection Refinement] Critique: {critique}. Auto-revising the final report..."
            })

        context = " ".join(self.gpt_researcher.context) if hasattr(self.gpt_researcher, 'context') else ""

        refinement_prompt = f"""You are an expert medical writer. You previously wrote a draft medical report, but a peer reviewer raised the following feedback and missing key points:

        [Reviewer Feedback]: {critique}
        [Missing Aspects]: {missing_aspects}

        Based on the reference materials provided, address all feedback, fill in the missing knowledge gaps, and restructure the report into a highly comprehensive, precise, and professional final version.

        Reference Context:
        {context[:4000]}

        Draft Report:
        {draft_report}

        Please return ONLY the revised final medical report in clean Markdown format:
        """
        try:
            # response = await self.gpt_researcher.llm.ainvoke(refinement_prompt)
            # final_report = response.content if hasattr(response, 'content') else str(response)
            messages = [{"role": "user", "content": refinement_prompt}]
            response_text = await create_chat_completion(
                messages=messages,
                model=self.gpt_researcher.cfg.smart_llm_model,
                temperature=0.4,
                llm_provider=self.gpt_researcher.cfg.smart_llm_provider,
                cost_callback=self.gpt_researcher.add_costs,
            )
            final_report = response_text
            return final_report
        except Exception as e:
            print(f"⚠️ [Reflection Refine Error]: {e}. Falling back to draft report.")
            return draft_report

    async def run(self):
        # 1. Conduct research (Supports earlier Hybrid + HITL flow)
        await self.gpt_researcher.conduct_research()
        
        # 2. Generate initial draft
        draft_report = await self.gpt_researcher.write_report()
        
        # 3. Reflection review layer
        review = await self._reflect_on_report(draft_report)
        
        # 4. Conditional refinement
        if not review.get("is_sufficient", True):
            final_report = await self._refine_report(draft_report, review)
            return final_report
            
        return draft_report

    async def run(self):
        #1.检索
        await self.gpt_researcher.conduct_research()
        #2.生成初稿
        draft_report = await self.gpt_researcher.write_report()
        # 3. 反思审查
        review = await self._reflect_on_report(draft_report)
        
        # 4. 条件修正
        if not review.get("is_sufficient", True):
            final_report = await self._refine_report(draft_report, review)
            return final_report

        return draft_report
