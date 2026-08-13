import asyncio
import hashlib
import json
import re
import time
from typing import List, Dict, Set, Optional, Any
from fastapi import WebSocket

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.llm import create_chat_completion


class DetailedReport:
    def __init__(
        self,
        query: str,
        report_type: str,
        report_source: str,
        source_urls: List[str] = [],
        document_urls: List[str] = [],
        query_domains: List[str] = [],
        config_path: str = None,
        tone: Any = "",
        websocket: WebSocket = None,
        subtopics: List[Dict] = [],
        headers: Optional[Dict] = None,
        complement_source_urls: bool = False,
        mcp_configs=None,
        mcp_strategy=None,
        max_search_results=None,
    ):
        self.query = query
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.query_domains = query_domains
        self.config_path = config_path
        self.tone = tone
        self.websocket = websocket
        self.subtopics = subtopics
        self.headers = headers or {}
        self.complement_source_urls = complement_source_urls
        self.max_search_results = max_search_results
        
        # Generate a unique research ID for this report
        self.research_id = self._generate_research_id(query)
        
        # Initialize researcher with optional MCP parameters
        gpt_researcher_params = {
            "query": self.query,
            "query_domains": self.query_domains,
            "report_type": "research_report",
            "report_source": self.report_source,
            "source_urls": self.source_urls,
            "document_urls": self.document_urls,
            "config_path": self.config_path,
            "tone": self.tone,
            "websocket": self.websocket,
            "headers": self.headers,
            "complement_source_urls": self.complement_source_urls,
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
        self.existing_headers: List[Dict] = []
        self.global_context: List[str] = []
        self.global_written_sections: List[str] = []
        self.global_urls: Set[str] = set(
            self.source_urls) if self.source_urls else set()

    def _generate_research_id(self, query: str) -> str:
        """Generate a unique research ID from query and timestamp."""
        timestamp = str(int(time.time()))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"detailed_{timestamp}_{query_hash}"

    async def run(self) -> str:
        # 1. 基础多子话题检索与生成
        await self._initial_research()
        subtopics = await self._get_all_subtopics()
        #用于快速测试detailedReport的Reflection
        subtopics = subtopics[:1]
        report_introduction = await self.gpt_researcher.write_introduction()
        _, report_body = await self._generate_subtopic_reports(subtopics)
        self.gpt_researcher.visited_urls.update(self.global_urls)
        
        # 2. 拼接完整的详细报告初稿
        draft_report = await self._construct_detailed_report(report_introduction, report_body)
        
        # 3. Reflection 反思与评审
        review = await self._reflect_on_report(draft_report)
        
        # 4. 根据评审结果决定是否精炼重写
        if not review.get("is_sufficient", True):
            final_report = await self._refine_report(draft_report, review)
            return final_report
            
        return draft_report

    async def _initial_research(self) -> None:
        await self.gpt_researcher.conduct_research()
        self.global_context = self.gpt_researcher.context
        self.global_urls = self.gpt_researcher.visited_urls

    async def _get_all_subtopics(self) -> List[Dict]:
        subtopics_data = await self.gpt_researcher.get_subtopics()

        all_subtopics = []
        if subtopics_data and subtopics_data.subtopics:
            for subtopic in subtopics_data.subtopics:
                all_subtopics.append({"task": subtopic.task})
        else:
            print(f"Unexpected subtopics data format: {subtopics_data}")

        return all_subtopics

    async def _generate_subtopic_reports(self, subtopics: List[Dict]) -> tuple:
        subtopic_reports = []
        subtopics_report_body = ""

        for subtopic in subtopics:
            result = await self._get_subtopic_report(subtopic)
            if result["report"]:
                subtopic_reports.append(result)
                subtopics_report_body += f"\n\n\n{result['report']}"

        return subtopic_reports, subtopics_report_body

    def _hashable_context(self, input_context: List[str] | List[dict]):
        context_items = []
        for item in input_context:
            if isinstance(item, dict):
                title = item.get("title", "No title")
                content = item.get("body", item.get("content", ""))
                context_str = f"Title: {title}\nContent: {content}"
                context_items.append(context_str)
            else:
                context_items.append(str(item))
        return context_items

    async def _get_subtopic_report(self, subtopic: Dict) -> Dict[str, str]:
        current_subtopic_task = subtopic.get("task")
        subtopic_assistant = GPTResearcher(
            query=current_subtopic_task,
            query_domains=self.query_domains,
            report_type="subtopic_report",
            report_source=self.report_source,
            websocket=self.websocket,
            headers=self.headers,
            parent_query=self.query,
            subtopics=self.subtopics,
            visited_urls=self.global_urls,
            agent=self.gpt_researcher.agent,
            role=self.gpt_researcher.role,
            tone=self.tone,
            complement_source_urls=self.complement_source_urls,
            source_urls=self.source_urls,
            mcp_configs=self.gpt_researcher.mcp_configs,
            mcp_strategy=self.gpt_researcher.mcp_strategy
        )

        if self.max_search_results is not None:
            subtopic_assistant.cfg.max_search_results_per_query = int(self.max_search_results)

        subtopic_assistant.context = list(set(self._hashable_context(self.global_context)))
        await subtopic_assistant.conduct_research()

        draft_section_titles = await subtopic_assistant.get_draft_section_titles(current_subtopic_task)

        if not isinstance(draft_section_titles, str):
            draft_section_titles = str(draft_section_titles)

        parse_draft_section_titles = self.gpt_researcher.extract_headers(draft_section_titles)
        parse_draft_section_titles_text = [header.get(
            "text", "") for header in parse_draft_section_titles]

        relevant_contents = await subtopic_assistant.get_similar_written_contents_by_draft_section_titles(
            current_subtopic_task, parse_draft_section_titles_text, self.global_written_sections
        )

        subtopic_report = await subtopic_assistant.write_report(
            existing_headers=self.existing_headers,
            relevant_written_contents=relevant_contents,
        )

        self.global_written_sections.extend(self.gpt_researcher.extract_sections(subtopic_report))
        self.global_context = list(set(self._hashable_context(subtopic_assistant.context)))
        self.global_urls.update(subtopic_assistant.visited_urls)

        self.existing_headers.append({
            "subtopic task": current_subtopic_task,
            "headers": self.gpt_researcher.extract_headers(subtopic_report),
        })

        return {"topic": subtopic, "report": subtopic_report}

    async def _construct_detailed_report(self, introduction: str, report_body: str) -> str:
        toc = self.gpt_researcher.table_of_contents(report_body)
        conclusion = await self.gpt_researcher.write_report_conclusion(report_body)
        conclusion_with_references = self.gpt_researcher.add_references(
            conclusion, self.gpt_researcher.visited_urls)
        report = f"{introduction}\n\n{toc}\n\n{report_body}\n\n{conclusion_with_references}"
        return report

    # ==================== Reflection 扩展模块 ====================

    async def _reflect_on_report(self, draft_report: str) -> Dict[str, Any]:
        """对深度医学报告初稿进行评审打分与诊断（极速纯文本解析版）"""
        print("\n🤔 [Reflection] Starting evaluation on detailed report...")
        
        if self.websocket:
            await self.websocket.send_json({
                "type": "logs",
                "output": "🤔 [Reflection] Reviewing detailed medical report..."
            })

        # 只截取前 2000 字符的大纲/开头
        safe_preview = draft_report[:2000]

        reflection_prompt = f"""You are a senior medical editor evaluating a report draft for query: "{self.query}".

Draft Preview:
{safe_preview}

Evaluate the quality and give a score from 1 to 10 based on completeness and clinical structure.
Please respond in EXACTLY this format:
SCORE: <number 1-10>
CRITIQUE: <one concise sentence review>
"""

        try:
            messages = [{"role": "user", "content": reflection_prompt}]
            
            # 调用 LLM
            response_text = await create_chat_completion(
                messages=messages,
                model=self.gpt_researcher.cfg.smart_llm_model,
                temperature=0.1,
                llm_provider=self.gpt_researcher.cfg.smart_llm_provider,
                cost_callback=self.gpt_researcher.add_costs,
            )

            # 兜底校验
            if not response_text or not response_text.strip():
                print("⚠️ [Reflection Warning]: LLM returned empty response. Falling back to default (Pass).")
                return {"score": 8, "is_sufficient": True, "critique": "Fallback due to empty LLM response"}

            print(f"📝 [LLM Raw Response]:\n{response_text}")

            # 使用正则解析 SCORE
            score_match = re.search(r"SCORE:\s*(\d+)", response_text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 8

            # 使用正则解析 CRITIQUE
            critique_match = re.search(r"CRITIQUE:\s*(.*)", response_text, re.IGNORECASE)
            critique = critique_match.group(1).strip() if critique_match else "Report looks acceptable."

            is_sufficient = score >= 8

            # 终端强行打印结果
            print(f"\n================ [Reflection Evaluation] ================")
            print(f"📊 Score: {score}/10 | Sufficient: {is_sufficient}")
            print(f"💬 Critique: {critique}")
            print(f"==========================================================\n")

            if self.websocket:
                await self.websocket.send_json({
                    "type": "logs",
                    "output": f"📊 [Reflection Score]: {score}/10 | Sufficient: {is_sufficient}\n💬 Critique: {critique}"
                })

            return {
                "score": score,
                "is_sufficient": is_sufficient,
                "critique": critique,
                "actionable_suggestions": [critique]
            }

        except Exception as e:
            print(f"⚠️ [Reflection Parser Error]: {e}")
            return {"score": 8, "is_sufficient": True, "critique": f"Skipped due to error: {e}"}

    async def _refine_report(self, draft_report: str, review: Dict[str, Any]) -> str:
        """根据审查意见对深度报告进行定点补充与精炼，避免大模型整体重写超时"""
        critique = review.get("critique", "")
        suggestions = "\n".join(review.get("actionable_suggestions", []))

        print("\n✏️ [Reflection Refinement] Generating targeted expert addendum/improvements...")

        if self.websocket:
            await self.websocket.send_json({
                "type": "logs",
                "output": "✏️ [Reflection Refinement] Refining detailed report based on peer review..."
            })

        # 为了防止传入完整报告导致输出 Token 溢出，只取核心上下文与评价
        refinement_prompt = f"""You are a senior medical expert. A peer review of our detailed literature report on "{self.query}" identified key areas for improvement.

    Reviewer Critique:
    {critique}

    Suggestions:
    {suggestions}

    Instructions:
    1. Write a high-quality, clinical-grade "Expert Clinical Addendum & Key Refinements" section that addresses all gaps mentioned in the critique.
    2. Ensure you provide structured details (e.g., specific age groups, warning signs, pediatric vs. adult presentation, or risk stratifications) as requested by the reviewer.
    3. Use Markdown formatting with clear subheadings.
    4. Output ONLY the new refinement/addendum content, which will be appended to strengthen the report.
    """
        try:
            messages = [{"role": "user", "content": refinement_prompt}]
            addendum = await create_chat_completion(
                messages=messages,
                model=self.gpt_researcher.cfg.smart_llm_model,
                temperature=0.3,
                llm_provider=self.gpt_researcher.cfg.smart_llm_provider,
                cost_callback=self.gpt_researcher.add_costs,
            )

            if not addendum or not addendum.strip():
                print("⚠️ [Refinement Warning]: Empty response received during refinement, returning draft.")
                return draft_report

            # 将补充修改内容无缝融合到报告正文中（插入在结论/参考文献之前，或作为专家补充章节）
            refined_final_report = f"""{draft_report}

        ---

        ## 🩺 Peer Review Refinements & Clinical Addendum

        > **Editorial Note:** This section was automatically generated by the LLM Reflection Module to address specific coverage gaps identified during medical peer review.

        {addendum.strip()}
        """

            print("✅ [Reflection Refinement] Enhanced report with Expert Clinical Addendum successfully!")

            if self.websocket:
                await self.websocket.send_json({
                    "type": "logs",
                    "output": "✅ Detailed Report refinement complete!"
                })

            return refined_final_report

        except Exception as e:
            print(f"⚠️ [Refinement Error]: {e}")
            return draft_report