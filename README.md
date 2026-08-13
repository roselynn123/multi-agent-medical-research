<div align="center" id="top">

<img src="https://github.com/assafelovic/gpt-researcher/assets/13554167/20af8286-b386-44a5-9a83-3be1365139c3" alt="Logo" width="80">


</div>

# 🔎 GPT Researcher


## Architecture

The core idea is to utilize 'planner' and 'execution' agents. The planner generates research questions, while the execution agents gather relevant information. The publisher then aggregates all findings into a comprehensive report.

<div align="center">
<img align="center" height="600" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527">
</div>

## Steps

1. **Task-specific research workflow**  
   Create a task-specific agent based on a research query. Generate sub-questions that collectively support an objective opinion; use crawler agents to gather evidence for each question; summarize and source-track every resource; then filter and aggregate the evidence into a final research report.

2. **Dynamic Plan-and-Execute + ReAct control**  
   Use a Plan-and-Execute architecture for top-level orchestration. A Planner (GPT-4o) decomposes complex medical topics into structured sub-queries, while parallel Executors (GPT-4o-mini) apply a ReAct feedback loop to assess retrieved evidence and dynamically decide whether to perform gap-filling retrieval or stop early.

3. **Three-mode medical retrieval engine: Web / Local / Hybrid**  
   Build a dual-track retrieval pipeline using fixed-size chunking with overlap, ChromaDB vector search, BM25 ranking, and Reciprocal Rank Fusion (RRF). Support web-only, local medical-library-only, and hybrid Web + Local retrieval modes.

4. **Strategic MCP protocol integration**  
   Integrate MCP protocol servers to connect local medical literature collections with external databases. Support `fast` retrieval through cache reuse and `deep` retrieval through per-sub-query exploration, enabling streamed fusion of authoritative literature and real-time web data.

5. **LLM-as-a-Judge evaluation and reflection**  
   After generating a report, use a prompt-driven LLM-as-a-Judge to evaluate its quality. When the score is below a configured threshold, automatically invoke a Refine Agent, using the judge’s reasoning as context for targeted correction and report reconstruction.

6. **WebSocket-based human-in-the-loop control**  
   After the Planner generates research sub-queries, inject a blocking signal and suspend backend execution using WebSocket communication and `asyncio.Event`. Researchers can add, edit, delete, or review sub-queries from the frontend and resume execution in real time, keeping clinical and pharmacological research on track.



## 📖 Documentation

See the [Documentation](https://docs.gptr.dev/docs/gpt-researcher/getting-started) for:
- Installation and setup guides
- Configuration and customization options
- How-To examples
- Full API references

<p align="right">
  <a href="#top">⬆️ Back to Top</a>
</p>
