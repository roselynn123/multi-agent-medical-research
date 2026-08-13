"""Research conductor skill for GPT Researcher.

This module provides the ResearchConductor class that manages and
coordinates the research process including query planning, web searching,
and context gathering.
"""

import asyncio
import logging
import os
import random

from ..actions.agent_creator import choose_agent
from ..actions.query_processing import get_search_results, plan_research_outline
from ..actions.utils import stream_output
from ..document import DocumentLoader, LangChainDocumentLoader, OnlineDocumentLoader
from ..utils.enum import ReportSource, ReportType
from ..utils.logging_config import get_json_handler
from medical_hybrid_store import load_hybrid_retriever
from gpt_researcher.utils.llm import create_chat_completion
import json

#from gpt_researcher.utils.enum import StreamType


class ResearchConductor:
    """Manages and coordinates the research process.

    This class handles the main research workflow including planning
    research queries, conducting web searches, managing MCP retrievers,
    and gathering context from various sources.

    Attributes:
        researcher: The parent GPTResearcher instance.
        logger: Logger for research events.
        json_handler: Handler for JSON logging.
    """

    def __init__(self, researcher):
        """Initialize the ResearchConductor.

        Args:
            researcher: The GPTResearcher instance that owns this conductor.
        """
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # Add cache for MCP results to avoid redundant calls
        self._mcp_results_cache = None
        # Guards cache population when research passes run concurrently
        self._mcp_cache_lock = asyncio.Lock()
        # Track MCP query count for balanced mode
        self._mcp_query_count = 0

    async def plan_research(self, query, query_domains=None):
        """Gets the sub-queries from the query
        Args:
            query: original query
        Returns:
            List of queries
        """
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        search_results = await get_search_results(
            query,
            self.researcher.retrievers[0],
            query_domains,
            researcher=self.researcher,
            max_results=self.researcher.cfg.max_search_results_per_query,
        )
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # Remove duplicate logging - this will be logged once in conduct_research instead

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # Pass retriever names for MCP optimization
            **self.researcher.kwargs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # Log active retrievers once at the start of research
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")
        
        # Note: visited_urls is deliberately NOT cleared here. It may be
        # shared with a parent researcher (e.g. detailed reports pass their
        # accumulated URLs into each subtopic researcher) so that already
        # scraped URLs are not fetched again.
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # Choose agent and role if not already defined
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )
                
        # Check if MCP retrievers are configured
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # Conduct research based on the source type
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)
        
        #原始的web搜索
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        
        
        elif self.researcher.report_source == ReportSource.Local.value:
            '''
            #原始的my_documents
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains) 
            '''
            self.logger.info("📚 [Local Pure Mode] 启动纯本地医学数据库 (Chroma+BM25) 检索模式")
            
            # 1. 加载我们本地构建好的 Chroma + BM25 混合检索器
            #from medical_hybrid_store import load_hybrid_retriever
            medical_retriever = load_hybrid_retriever()
            
            if not medical_retriever:
                self.logger.warning("⚠️ 未检测到已构建的本地数据库，请先运行建库脚本！")
                research_data = "未找到本地医学文献数据。"
            else:
                # 2. 生成 Planner 的子问题（借用 llm 规划子问题，但绝不联网搜网页）
                sub_queries = await self.plan_research(self.researcher.query, self.researcher.query_domains)

                self.logger.info(f"Generated sub-queries: {sub_queries}")
                '''
                # ==================== 🛑 [HITL 人工干预拦截器] ====================
                print("\n" + "═"*60)
                print("🛑 [Human-in-the-Loop] Agent 规划了以下研究子问题：")
                for idx, q in enumerate(sub_queries, 1):
                    print(f"  {idx}. {q}")
                print("═"*60)

                # 提示用户进行操作
                print("\n👉 请选择操作:")
                print("  [回车] 保持默认，直接继续")
                print("  [a]   添加新子问题 (Add)")
                print("  [d]   删除指定子问题 (Delete)")
                print("  [m]   重新输入整套子问题 (Modify)")
                
                user_action = input("\n请输入指令 (Enter/a/d/m): ").strip().lower()

                if user_action == 'a':
                    add_q = input("请输入要追加的子问题: ").strip()
                    if add_q:
                        sub_queries.append(add_q)
                        print(f"✅ 已添加！最新子问题列表: {sub_queries}")

                elif user_action == 'd':
                    del_idx = input("请输入要删除的子问题编号 (例如 1 或 1,2): ").strip()
                    try:
                        # 解析要删除的序号并降序排列删除，防止 index 错乱
                        indices_to_del = sorted([int(x.strip()) - 1 for x in del_idx.split(",")], reverse=True)
                        for idx in indices_to_del:
                            if 0 <= idx < len(sub_queries):
                                removed = sub_queries.pop(idx)
                                print(f"🗑️ 已删除: {removed}")
                        print(f"✅ 更新后的子问题列表: {sub_queries}")
                    except Exception as e:
                        print(f"⚠️ 输入格式错误，保持原有子问题: {e}")

                elif user_action == 'm':
                    print("请输入完整的子问题列表 (多个子问题请用英文逗号 ',' 分隔):")
                    new_queries_str = input("> ").strip()
                    if new_queries_str:
                        sub_queries = [q.strip() for q in new_queries_str.split(",") if q.strip()]
                        print(f"✅ 已彻底重写！最新子问题列表: {sub_queries}")
                
                else:
                    print("▶️ 保持默认规划，继续执行...")
                
                print("="*60 + "\n")
                # ==================================================================
                '''
                #-----------------------------Human-in-the-loop-------------------------------------------------
                if self.researcher.websocket:
                    # 1. 创建一个异步锁/事件信号
                    from server.server_utils import hitl_feedback_event,hitl_user_data

                    # 1. 向前端推送拦截请求 (带上当前的 sub_queries)
                    await self.researcher.websocket.send_json({
                        "type": "human_feedback_request",
                        "content": "subqueries_review",
                        "output": {
                            "sub_queries": sub_queries
                        }
                    })
                    
                    self.logger.info("⏳ [HITL] 已向前端发送干预请求，Agent 暂停等待用户操作...")

                    # 2. 阻塞挂起，等待 server_utils 收到 human_feedback 指令后触发 set()
                    await hitl_feedback_event.wait()

                    # 3. 拿到用户从前端修改后的 sub_queries 并覆盖
                    if "sub_queries" in hitl_user_data and isinstance(hitl_user_data["sub_queries"], list):
                        sub_queries = hitl_user_data["sub_queries"]
                        self.logger.info(f"✅ [HITL] 成功获取前端干预数据，更新后的子问题: {sub_queries}")
                    
                    # 清空 Event 给下一次使用
                    hitl_feedback_event.clear()
                #-------------------------------------------------------------------------------
                if self.researcher.report_type != "subtopic_report":
                    sub_queries.append(self.researcher.query)
                
                self.logger.info(f"🗂️ 本地检索将基于以下子问题提取文献: {sub_queries}")
                
                # 3. 遍历所有子问题，在本地 Chroma+BM25 数据库中提取 Context
                combined_local_contexts = []
                seen_contents = set()  # 用于文本去重
                lock=asyncio.Lock()
                '''
                print("\n" + "="*60)
                print(f"📖 [Local Pure Search] 开始在本地医学库中检索 {len(sub_queries)} 个子任务...")
                
                for q in sub_queries:
                    docs = medical_retriever.invoke(q)
                    for d in docs:
                        # 简单的去重逻辑，防止重复切片占用 Context
                        content_snippet = d.page_content[:100]
                        if content_snippet not in seen_contents:
                            seen_contents.add(content_snippet)
                            source = d.metadata.get('source', '本地文献')
                            page = d.metadata.get('page', '0')
                            
                            formatted_doc = f"[本地文献: {source} (第{page}页)]\n{d.page_content}"
                            combined_local_contexts.append(formatted_doc)
                            print(f"  └─ 命中文献: {source} (P.{page}) | 查询: '{q}'")
                '''
                print("\n" + "="*60)
                print(f"📖 [Local ReAct Search] Launching ReAct agents across {len(sub_queries)} sub-tasks...")
                print("="*60)

                # Execute ReAct sub-agents concurrently using asyncio.gather
                tasks = [
                    self._react_local_search_subquery(
                        sub_query=q, 
                        medical_retriever=medical_retriever, 
                        seen_contents=seen_contents,
                        lock=lock,
                        max_steps=3  # Maximum 2 ReAct iterations per sub-query
                    ) 
                    for q in sub_queries
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Aggregate all retrieved contexts
                for res in results:
                    combined_local_contexts.extend(res)
                print("="*60 + "\n")
                
                # 4. 拼装纯本地的 Context 数据
                research_data = "\n\n".join(combined_local_contexts)
                self.logger.info(f"纯本地检索完成，共提取 {len(combined_local_contexts)} 条本地文献片段。")


        # Hybrid search including both local documents and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            self.logger.info("🔬 [Hybrid Mode] 正在读取本地医学数据库 (Chroma+BM25)...")
            # 1. 加载我们本地构建好的 Chroma + BM25 混合检索器
            #from medical_hybrid_store import load_hybrid_retriever
            medical_retriever = load_hybrid_retriever()
            # 2. 包装成一个包含 retriever 的 list 作为第二个参数
            # 这样既满足了 list 类型要求，又把 retriever 实体带进了 ReAct 内部
            local_data_param = [{"type": "custom_hybrid_retriever", "retriever": medical_retriever}]

            self.logger.info("🚀 [Hybrid Mode] 启动 [Local DB ReAct] 与 [Web ReAct] 双轨并发检索...")

            # 3. 双轨并发调用 _get_context_by_web_search
            # 轨一：传入 local_data_param (非空 list) -> 触发本地 CustomHybridRetriever 检索
            # 轨二：传入 []                (空 list)   -> 触发 Web 搜索引擎检索
            docs_context, web_context = await asyncio.gather(
                self._get_context_by_web_search(self.researcher.query, local_data_param, self.researcher.query_domains),
                self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains),
            )

            # 4. 知识融合
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)

            '''
            # 1. 先让 Web Search 去生成 Planner 的子问题
            # 注意：这里我们传入一个空的 sub_queries 拦截逻辑
            #web_context = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            web_context, approved_sub_queries = await self._get_context_by_web_search(
                self.researcher.query, [], self.researcher.query_domains
            )

            # 2. 专门针对主问题及所有相关问题，查询本地 Chroma+BM25 库
            async def _get_local_hybrid_context():
                if not medical_retriever:
                    self.logger.warning("⚠️ 本地医学数据库未就绪，跳过本地检索。")
                    return ""
                
                # 查本地库
                docs = medical_retriever.invoke(self.researcher.query)
                
                print("\n" + "="*50)
                print(f"📚 [Local Hybrid DB] 为查询 '{self.researcher.query}' 检索到 {len(docs)} 条本地医学文献片段：")
                context_chunks = []
                for i, d in enumerate(docs, 1):
                    source = d.metadata.get('source', '未知文件')
                    page = d.metadata.get('page', '0')
                    snippet = d.page_content[:150].replace('\n', ' ')
                    print(f"  └─ [{i}] 来源: {source} (P.{page}) | 预览: {snippet}...")
                    context_chunks.append(f"[本地医学文献: {source} (第{page}页)]\n{d.page_content}")
                print("="*50 + "\n")
                
                return "\n\n".join(context_chunks)

            docs_context = await _get_local_hybrid_context()
            
            # 3. 融合本地文献和网络文献
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)

            
            #原本的hybird-----------------------------------------------------------------------------
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            # The local-docs pass and the web pass are independent, so run
            # them concurrently; visited_urls still dedupes across both.
            docs_context, web_context = await asyncio.gather(
                self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains),
                self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains),
            )
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
            #--------------------------------------------------------------------------------------------------
            '''

        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)
            
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # Rank and curate the sources
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            curated = await self.researcher.source_curator.curate_sources(research_data)
            # curate_sources() returns List[dict] with Title/Content/Source keys.
            # Normalize to str so downstream code that expects researcher.context
            # to be a string (e.g. "\n".join, .split(), len()) doesn't crash.
            if isinstance(curated, list):
                self.researcher.context = "\n\n".join(
                    "Title: {title}\nContent: {content}\nSource: {source}".format(
                        title=s.get("Title", ""),
                        content=s.get("Content", ""),
                        source=s.get("Source", ""),
                    ) if isinstance(s, dict) else str(s)
                    for s in curated
                )
            else:
                self.researcher.context = curated

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # Add logging to other methods similarly...

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)
        
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context
    #使用web检索去生成subqueries
    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # 判断当前是【本地 DB 模式】还是【Web Search 模式】
        is_local_track = len(scraped_data) > 0 or any(
            "customhybrid" in getattr(r, "__name__", "").lower() or "local" in getattr(r, "__name__", "").lower()
            for r in getattr(self.researcher, "retrievers", [])
        )
        mode_label = "Local DB" if is_local_track else "Web Search"

        # **CONFIGURABLE MCP OPTIMIZATION: Control MCP strategy**
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
        mcp_strategy = self._get_mcp_strategy()
        
        async with self._mcp_cache_lock:
            if mcp_retrievers and self._mcp_results_cache is None:
                if mcp_strategy == "disabled":
                    self.logger.info("MCP disabled by strategy, skipping MCP research")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_disabled",
                            f"⚡ MCP research disabled by configuration",
                            self.researcher.websocket,
                        )
                elif mcp_strategy == "fast":
                    self.logger.info("MCP fast strategy: Running once with original query")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_optimization",
                            f"🚀 MCP Fast: Running once for main query (performance mode)",
                            self.researcher.websocket,
                        )
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
                elif mcp_strategy == "deep":
                    self.logger.info("MCP deep strategy: Will run for all queries")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive",
                            f"🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                            self.researcher.websocket,
                        )
                else:
                    self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                    mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                    self._mcp_results_cache = mcp_context
                    self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")

        # ==================== 🛑 [WebSocket HITL 人工干预拦截器] ====================
        if self.researcher.websocket:
            from server.server_utils import hitl_feedback_event, hitl_user_data

            # 1. 向前端推送带有 sub_queries 的弹窗请求
            await self.researcher.websocket.send_json({
                "type": "human_feedback_request",
                "content": "subqueries_review",
                "output": {
                    "sub_queries": sub_queries
                }
            })
            
            self.logger.info("⏳ [HITL - WebSearch] 已向前端发送干预请求，Agent 暂停等待用户操作...")

            # 2. 阻塞挂起，等待 server_utils 收到 WebSocket 的 human_feedback 指令
            await hitl_feedback_event.wait()

            # 3. 拿到用户从前端修改后的 sub_queries 彻底覆盖
            if "sub_queries" in hitl_user_data and isinstance(hitl_user_data["sub_queries"], list):
                sub_queries = list(hitl_user_data["sub_queries"])
                self.logger.info(f"✅ [HITL - WebSearch] 成功获取前端干预数据，更新后的子问题: {sub_queries}")
            
            # 4. 清空 Event 备用
            hitl_feedback_event.clear()
        # ===========================================================================

        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        # 🧹【防重优化】：对 sub_queries 去重，避免重复检索
        sub_queries = list(dict.fromkeys(sub_queries))

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # ==================== 🛠️ [ReAct 逻辑重构（彻底修复控制词误搜问题）] ====================
        import re

        def _clean_llm_query(query_text: str, default_query: str) -> str | None:
            """清洗与校验 LLM 返回的检索词"""
            cleaned = query_text.strip()
            
            # 剥离提示词污染前缀
            patterns = [
                r"^refined\s*search\s*query\s*(string)?:?\s*",
                r"^refined\s*search\s*:?\s*",
                r"^search\s*query:?\s*",
                r"^new\s*query:?\s*",
                r"^keywords?:?\s*",
            ]
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" '\"`\n\r")

            # 关键拦截：防止 LLM 把控制词（如 INSUFFICIENT）输出为搜索词
            blacklisted_words = {"INSUFFICIENT", "NOT ENOUGH", "NEED MORE", "NO", "FALSE", "MORE INFO NEEDED"}
            if cleaned.upper() in blacklisted_words or len(cleaned) < 3:
                return None  # 返回 None 表示提取无效，将终止或重试
                
            return cleaned

        async def _react_search_single_subquery(target_subquery: str, max_steps: int = 2) -> str:
            """独立的 async ReAct 子任务处理器"""
            current_query = target_subquery
            accumulated_chunks = []
            
            for step in range(1, max_steps + 1):
                # Action: 执行检索
                #context_chunk = await self._process_sub_query(current_query, scraped_data, query_domains)#原来的没有超时处理的代码
                # 带有 15 秒超时保护的子查询处理，防止某单个爬虫连接死锁拖垮整体
                try:
                    context_chunk = await asyncio.wait_for(
                        self._process_sub_query(current_query, scraped_data, query_domains),
                        timeout=25.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"⚠️ [{mode_label} ReAct] 执行子查询 '{current_query}' 超时(25s)，跳过该步")
                    context_chunk = ""

                if context_chunk:
                    accumulated_chunks.append(context_chunk)
                    
                if step == max_steps:
                    break
                    
                # Thought: 反思与评估
                reflection_prompt = f"""You are an expert AI Researcher examining retrieved information for [{mode_label}].

Sub-topic to Answer: "{target_subquery}"
Current Iteration Step: {step}/{max_steps}
Retrieved Context so far:
{context_chunk[:1000] if context_chunk else "No data retrieved."}

Task:
Determine if the retrieved context is sufficient to answer the sub-topic thoroughly.

Rules for Output:
1. If the context is SUFFICIENT: Output EXACTLY the single word 'ENOUGH'.
2. If the context is NOT SUFFICIENT: Output ONLY a specific, optimized search keyword phrase (e.g. "coronary artery disease chest pain symptoms") that will yield missing details.

CRITICAL INSTRUCTIONS:
- DO NOT write 'INSUFFICIENT', 'NOT ENOUGH', or any status label.
- DO NOT write markdown formatting or explanations.
- Output ONLY 'ENOUGH' OR the plain search keywords:"""

                try:
                    if hasattr(self, 'researcher') and hasattr(self.researcher, 'llm') and self.researcher.llm:
                        llm_instance = self.researcher.llm
                    else:
                        from langchain_openai import ChatOpenAI
                        llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                    response = await llm_instance.ainvoke(reflection_prompt)
                    raw_decision = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    
                    if raw_decision.upper() == 'ENOUGH' or 'ENOUGH' in raw_decision.upper():
                        self.logger.info(f"💡 [{mode_label} ReAct] 子问题 '{target_subquery}' 获得充足信息，提早结束 (Step {step})")
                        break
                    else:
                        cleaned_kw = _clean_llm_query(raw_decision, target_subquery)
                        if cleaned_kw:
                            current_query = cleaned_kw
                            self.logger.info(f"💡 [{mode_label} ReAct] Step {step} 信息不足。反思修正后的检索词: '{current_query}'")
                        else:
                            self.logger.warning(f"⚠️ [{mode_label} ReAct] Step {step} LLM输出了非法/非预期控制词 ('{raw_decision}')，已自动拦截，停止进一步无效检索。")
                            break
                except Exception as e:
                    self.logger.error(f"[{mode_label} ReAct] LLM reflect error: {e}")
                    break

            return "\n\n".join(accumulated_chunks)

        # 执行异步并发调度
        try:
            tasks = [_react_search_single_subquery(sq, max_steps=2) for sq in sub_queries]
            context = await asyncio.gather(*tasks)
            
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            context = [c for c in context if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return ""
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return ""

    def _get_mcp_strategy(self) -> str:
        """
        Get the MCP strategy configuration.
        
        Priority:
        1. Instance-level setting (self.researcher.mcp_strategy)
        2. Config file setting (self.researcher.cfg.mcp_strategy) 
        3. Default value ("fast")
        
        Returns:
            str: MCP strategy
                "disabled" = Skip MCP entirely
                "fast" = Run MCP once with original query (default)
                "deep" = Run MCP for all sub-queries
        """
        # Check instance-level setting first
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy
        
        # Check config setting
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy
        
        # Default to fast mode
        return "fast"

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        Execute MCP research for a list of queries.
        
        Args:
            queries: List of queries to research
            mcp_retrievers: List of MCP retriever classes
            
        Returns:
            list: Combined MCP context entries from all queries
        """
        all_mcp_context = []
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")
            
            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")
                            
                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)
                        
                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")
                        
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )
        
        return all_mcp_context

    def _tavily_mcp_redundant_with_direct(self, mcp_retrievers, non_mcp_retrievers) -> bool:
        """True when MCP would only re-query Tavily while direct Tavily is active.

        The frontend Tavily Web Search MCP preset hits the same API as
        `TavilySearch` and adds extra LLM tool-selection cost for no new data
        when both run together (#1875).
        """
        if not mcp_retrievers or not non_mcp_retrievers:
            return False
        has_direct_tavily = any(
            getattr(r, "__name__", "").lower() == "tavilysearch" for r in non_mcp_retrievers
        )
        if not has_direct_tavily:
            return False
        configs = getattr(self.researcher, "mcp_configs", None) or []
        if not configs:
            return False
        # If every configured MCP server is a Tavily MCP package, treat as redundant.
        def _is_tavily_mcp(cfg: dict) -> bool:
            name = str(cfg.get("name", "")).lower()
            args = " ".join(str(a) for a in (cfg.get("args") or [])).lower()
            command = str(cfg.get("command", "")).lower()
            blob = f"{name} {args} {command}"
            return "tavily" in blob

        return all(isinstance(c, dict) and _is_tavily_mcp(c) for c in configs)


    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # ==================== 💡 [新增：医学 Chroma+BM25 向量库拦截分支] ====================
            if scraped_data and isinstance(scraped_data, list) and len(scraped_data) > 0:
                first_item = scraped_data[0]
                # 识别出我们传入的 CustomHybridRetriever 包装对象
                if isinstance(first_item, dict) and first_item.get("type") == "custom_hybrid_retriever":
                    retriever = first_item.get("retriever")
                    if retriever:
                        self.logger.info(f"📚 [Local Hybrid DB] 正在使用 CustomHybridRetriever 检索: '{sub_query}'")
                        # 🎯 执行你的本地向量库 + BM25 检索
                        docs = retriever.invoke(sub_query)
                        
                        # 将 List[Document] 转成原生 context_manager 识别的文本块字典格式
                        scraped_data = [
                            {
                                "raw_content": doc.page_content,
                                "url": f"local_pdf://{doc.metadata.get('source', 'unknown')}#page={doc.metadata.get('page', 0)}"
                            }
                            for doc in docs
                        ]
            # ====================================================================================


            # Identify MCP retrievers
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]

            # Avoid dual Tavily path (direct retriever + tavily-mcp) under default RETRIEVER=tavily.
            if self._tavily_mcp_redundant_with_direct(mcp_retrievers, non_mcp_retrievers):
                self.logger.warning(
                    "Skipping LLM MCP Tavily path because TavilySearch is already configured as a direct retriever; set RETRIEVER without tavily or use non-Tavily MCP servers to keep MCP."
                )
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_tavily_deduped",
                        "⚠️ Skipping Tavily MCP (redundant with direct Tavily retriever) to avoid double API cost",
                        self.researcher.websocket,
                    )
                mcp_retrievers = []
            
            # Initialize context components
            mcp_context = []
            web_context = ""
            
            # Get MCP strategy configuration
            mcp_strategy = self._get_mcp_strategy()
            
            # **CONFIGURABLE MCP PROCESSING**
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP disabled - skip entirely
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # Fast: Use cached results
                    mcp_context = self._mcp_results_cache.copy()
                    
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # Deep: Run MCP for every sub-query
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # Fallback: if no cache and not deep mode, run MCP for this query
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )
                    
                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
            
            # Get web search context using non-MCP retrievers (if no scraped data provided)
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # Get similar content based on scraped data
            if scraped_data:
                web_context = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # Combine MCP context with web context intelligently
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)
            
            # Log context combination results
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")
                
                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )
            
            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })
                
            return combined_context
            
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {str(e)}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        Execute MCP research using the new two-stage approach.
        
        Args:
            retriever: The MCP retriever class
            query: The search query
            
        Returns:
            list: MCP research results
        """
        retriever_name = retriever.__name__
        
        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the MCP retriever with proper parameters
            # Pass the researcher instance (self.researcher) which contains both cfg and mcp_configs
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # Pass the entire researcher instance
            )
            
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )
            
            # Execute the two-stage MCP search
            results = retriever_instance.search(
                max_results=self.researcher.cfg.max_search_results_per_query
            )
            
            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")
                
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )
                
                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []
                
        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {str(e)}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {str(e)} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        Intelligently combine MCP and web research context.
        
        Args:
            mcp_context: List of MCP context entries
            web_context: Web research context string  
            sub_query: The sub-query being processed
            
        Returns:
            str: Combined context string
        """
        combined_parts = []
        
        # Add web context first if available
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")
        
        # Add MCP context with proper formatting
        if mcp_context:
            mcp_formatted = []
            
            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")
                
                if content and content.strip():
                    # Create a well-formatted context entry
                    if url and url != f"mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"
                    
                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)
            
            if mcp_formatted:
                # Join MCP results with clear separation
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")
        
        # Combine all parts
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None):
        new_search_urls = []
        prefetched_content = []
        if query_domains is None:
            query_domains = []

        # Iterate through the currently set retrievers
        # This allows the method to work when retrievers are temporarily modified
        for retriever_class in self.researcher.retrievers:
            # Skip MCP retrievers as they don't provide URLs for scraping
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # Instantiate the retriever with the sub-query
                retriever = retriever_class(query, query_domains=query_domains)

                # Perform the search using the current retriever
                search_results = await asyncio.to_thread(
                    retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                )

                if not search_results:
                    continue

                # Separate results that already have content from those needing scraping
                for result in search_results:
                    url = result.get("href") or result.get("url")
                    raw_content = result.get("raw_content")
                    if url and raw_content and len(raw_content) > 100:
                        # Only raw_content signals that a retriever already fetched the full page.
                        # body is snippet-sized text for most web retrievers and still needs scraping.
                        prefetched_content.append({
                            "url": url,
                            "raw_content": raw_content,
                        })
                        self.researcher.add_research_sources([{"url": url}])
                    elif url:
                        new_search_urls.append(url)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")

        # Get unique URLs
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls, prefetched_content

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.
        Retrievers that already provide full content (e.g. PubMed Central) have their
        content passed through directly without re-scraping.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        if query_domains is None:
            query_domains = []

        new_search_urls, prefetched_content = await self._search_relevant_source_urls(sub_query, query_domains)

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape URLs that need fetching (skip those already provided by retrievers)
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        # Merge pre-fetched content from retrievers that already provide full text
        scraped_content.extend(prefetched_content)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _search(self, retriever, query):
        """
        Perform a search using the specified retriever.
        
        Args:
            retriever: The retriever class to use
            query: The search query
            
        Returns:
            list: Search results
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()
        
        self.logger.info(f"Searching with {retriever_name} for query: {query}")
        
        try:
            # Instantiate the retriever
            retriever_instance = retriever(
                query=query, 
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )
            
            # Log MCP server configurations if using MCP retriever
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )
            
            # Perform the search
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )
                
                # Log result information
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")
                    
                    # Special logging for MCP retriever
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )
                        
                        # Log result details
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")
                            
                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )
                
                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {str(e)}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {str(e)}",
                    self.researcher.websocket,
                )
            return []
            
    async def _extract_content(self, results):
        """
        Extract content from search results using the browser manager.
        
        Args:
            results: Search results
            
        Returns:
            list: Extracted content
        """
        self.logger.info(f"Extracting content from {len(results)} search results")
        
        # Get the URLs from the search results
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])
        
        # Skip if no URLs found
        if not urls:
            return []
            
        # Make sure we don't visit URLs we've already visited
        new_urls = [url for url in urls if url not in self.researcher.visited_urls]
        
        # Return empty if no new URLs
        if not new_urls:
            return []
            
        # Scrape the content from the URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)
        
        # Add the URLs to visited_urls
        self.researcher.visited_urls.update(new_urls)
        
        return scraped_content
        
    async def _summarize_content(self, query, content):
        """
        Summarize the extracted content.
        
        Args:
            query: The search query
            content: The extracted content
            
        Returns:
            str: Summarized content
        """
        self.logger.info(f"Summarizing content for query: {query}")
        
        # Skip if no content
        if not content:
            return ""
            
        # Summarize the content using the context manager
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )
        
        return summary
        
    async def _update_search_progress(self, current, total):
        """
        Update the search progress.
        
        Args:
            current: Current number of sub-queries processed
            total: Total number of sub-queries
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )

    #本地向量数据库中的ReAct
    async def _react_local_search_subquery(self, sub_query: str, medical_retriever, seen_contents: set,lock: asyncio.Lock,max_steps: int = 2) -> list:
        """
        Executes a ReAct (Reasoning-Action-Observation) loop for a single sub-query 
        against the local medical retriever.
        """
        sub_contexts = []
        current_query = sub_query
        
        self.logger.info(f"🤖 [ReAct Agent] Processing sub-task: '{sub_query}'")

        for step in range(max_steps):
            print(f"  ├── 🔄 [ReAct Step {step + 1}/{max_steps}] Querying: '{current_query}'")
            
            # 1. Action & Observation: Retrieve docs from local hybrid database
            # If your retriever supports async, use: await medical_retriever.ainvoke(current_query)
            docs = medical_retriever.invoke(current_query)
            
            new_docs_added = 0
            for d in docs:
                content_snippet = d.page_content[:100]
                
                # Thread-safe duplicate check using Async Lock
                async with lock:
                    is_duplicate = content_snippet in seen_contents
                    if not is_duplicate:
                        seen_contents.add(content_snippet)

                if not is_duplicate:
                    source = d.metadata.get('source', 'Local Reference')
                    page = d.metadata.get('page', '0')
                    
                    formatted_doc = f"[Local Literature: {source} (Page {page})]\n{d.page_content}"
                    sub_contexts.append(formatted_doc)
                    new_docs_added += 1
                    print(f"  │    └─ Matched Doc: {source} (P.{page})")

            # If we reached the max steps, skip the reasoning LLM call to save time/tokens
            if step == max_steps - 1:
                break

            # 2. Reasoning: Prompt LLM to evaluate if retrieved evidence is sufficient
            reflection_prompt = f"""You are a rigorous medical researcher evaluating retrieved evidence for a specific research question.

        Sub-research task: "{sub_query}"

        Currently retrieved evidence snippets:
        {chr(10).join(sub_contexts) if sub_contexts else "No evidence retrieved yet."}

        Evaluation Requirement:
        Determine if the retrieved evidence above is sufficient, comprehensive, and detailed enough to fully answer the sub-research task: "{sub_query}".

        Instructions:
        - If the information is SUFFICIENT, output ONLY the string: ENOUGH
        - If key details are MISSING (e.g., specific dosage, mechanism, adverse effects, or clinical trials), generate a SINGLE optimized search query targeting the missing information. 
        - Output ONLY the refined query string without any explanations, markdown quotes, or additional text.
        """
            try:
                # Call the LLM (Adjust the invoke call based on your framework's wrapper)
                # response = await self.llm.ainvoke(reflection_prompt)
                # decision = response.content.strip()
                # 优先检查 self.researcher 是否提供 llm，如无则直接使用 LangChain 构造
                if hasattr(self, 'researcher') and hasattr(self.researcher, 'llm') and self.researcher.llm:
                    llm_instance = self.researcher.llm
                else:
                    from langchain_openai import ChatOpenAI
                    # 默认使用配置中的模型或轻量模型 gpt-4o-mini 做反思判断
                    llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                response = await llm_instance.ainvoke(reflection_prompt)
                decision = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # 3. Decision Assessment
                if "ENOUGH" in decision.upper():
                    print(f"  └── ✅ [ReAct Agent] Sufficient evidence found. Terminating ReAct loop early.")
                    break
                else:
                    # Clean up quotes if LLM outputs quotes around the refined query
                    current_query = decision.replace('"', '').replace("'", "").strip()
                    print(f"  ├── 💡 [ReAct Agent] Information insufficient. Refined search query: '{current_query}'")
            
            except Exception as e:
                self.logger.error(f"⚠️ Exception during ReAct reflection phase: {e}")
                break

        return sub_contexts
    
    #web检索时的ReAct
    async def _react_web_search_subquery(self, sub_query: str, scraped_data: list, query_domains: list, max_steps: int = 2) -> str:
        """
        网络端 ReAct 智能体：针对单个子问题，进行 Thought -> Action (Web Search & Scrape) -> Observation 循环
        """
        current_query = sub_query
        accumulated_context = []
        
        self.logger.info(f"🌐 [Web ReAct Agent] 启动子任务: '{sub_query}'")
        
        for step in range(1, max_steps + 1):
            # 1. 执行网络搜索与网页抓取 (复用原本的单次检索函数)
            context_chunk = await self._process_sub_query(current_query, scraped_data, query_domains)
            if context_chunk:
                accumulated_context.append(context_chunk)
                
            # 如果是最后一步，直接退出
            if step == max_steps:
                break
                
            # 2. ReAct Thought: 让 LLM 反思当前抓到的网页信息是否足够回答子问题
            prompt = f"""You are a Web Research Agent investigating a specific sub-topic.

        Original Sub-topic: "{sub_query}"
        Current Search Step: {step}/{max_steps}
        Retrieved Web Information so far:
        {context_chunk[:1000] if context_chunk else "No information retrieved yet."}

        Task:
        Analyze if the retrieved web information is sufficient to comprehensively answer the sub-topic.
        - If SUFFICIENT: Reply with EXACTLY 'ENOUGH'.
        - If INSUFFICIENT: Reply with a refined search query (a few keywords or a specific phrase) to search the web for missing details.

        Output ONLY 'ENOUGH' or the refined query string:"""

            try:
                # response = await self.researcher.llm.ainvoke(prompt)
                # decision = response.content.strip()
                if hasattr(self, 'researcher') and hasattr(self.researcher, 'llm') and self.researcher.llm:
                    llm_instance = self.researcher.llm
                else:
                    from langchain_openai import ChatOpenAI
                    # 默认使用配置中的模型或轻量模型 gpt-4o-mini 做反思判断
                    llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                response = await llm_instance.ainvoke(reflection_prompt)
                decision = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                if decision.upper() == 'ENOUGH' or 'ENOUGH' in decision.upper():
                    self.logger.info(f"💡 [Web ReAct Agent] 信息已充分，提早结束步数 (Step {step})")
                    break
                else:
                    # 拿 LLM 优化后的新关键词进行下一轮网络搜索
                    current_query = decision.replace('"', '').replace("'", "").strip()
                    self.logger.info(f"💡 [Web ReAct Agent] Step {step} 信息不足。优化后的网络搜索词: '{current_query}'")
            except Exception as e:
                self.logger.error(f"Web ReAct LLM reflect error: {e}")
                break

        return "\n\n".join(accumulated_context)