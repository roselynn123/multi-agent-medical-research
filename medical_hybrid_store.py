import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 1. 自定义一个简单可靠的混合检索器，摆脱 EnsembleRetriever 的版本导入困擾
class CustomHybridRetriever:
    def __init__(self, bm25_retriever, chroma_retriever, weight_bm25=0.5, weight_chroma=0.5):
        self.bm25 = bm25_retriever
        self.chroma = chroma_retriever
        self.w_bm25 = weight_bm25
        self.w_chroma = weight_chroma

    def invoke(self, query: str) -> List[Document]:
        # 分别调用两个检索器
        bm25_docs = self.bm25.invoke(query)
        chroma_docs = self.chroma.invoke(query)
        
        # 简单去重与合并 (优先保留排名前列的文档)
        seen_contents = set()
        combined_docs = []

        # 交替交叉合并结果 (RRF 思想的简化版)
        for b_doc, c_doc in zip(bm25_docs, chroma_docs):
            if b_doc.page_content not in seen_contents:
                seen_contents.add(b_doc.page_content)
                combined_docs.append(b_doc)
            if c_doc.page_content not in seen_contents:
                seen_contents.add(c_doc.page_content)
                combined_docs.append(c_doc)

        return combined_docs

# 2. 配置路径
PDF_DIR = "./medical_docs"
CHROMA_DB_DIR = "./medical_chroma_db"

def build_hybrid_retriever():
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        print(f"⚠️ 请先将医学 PDF 文件放入 '{PDF_DIR}' 目录下！")
        return None

    print("📄 1. 正在加载本地医学 PDF...")
    loader = PyPDFDirectoryLoader(PDF_DIR)
    documents = loader.load()
    print(f"   已加载 {len(documents)} 页文档。")

    print("✂️ 2. 正在进行文本切片 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)
    print(f"   切片完成，共生成 {len(docs)} 个文本块。")

    print("🧠 3. 构建向量数据库 (ChromaDB)...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("🔍 4. 构建关键词检索器 (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    print("🔀 5. 融合构建混合检索器...")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 使用自定义的混合检索器
    hybrid_retriever = CustomHybridRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever
    )
    
    print("✅ 本地医学混合检索数据库构建完成！")
    return hybrid_retriever

def query_medical_db(retriever, query: str):
    print(f"\n🔎 正在查询: '{query}'")
    results = retriever.invoke(query)
    
    print(f"找到了 {len(results)} 条最相关的上下文：")
    for idx, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        snippet = doc.page_content.replace("\n", " ")[:150]
        print(f"\n--- [{idx}] 来源: {source} (第 {page} 页) ---")
        print(f"{snippet}...")
        
def load_hybrid_retriever():
    """
    【在线检索用】不解析 PDF，直接加载磁盘上已建好的 Chroma 向量库 + 快速加载 BM25
    """
    if not os.path.exists(CHROMA_DB_DIR):
        print("⚠️ 磁盘上未找到已建好的 Chroma 数据库，请先运行建库脚本！")
        return None

    embeddings = OpenAIEmbeddings()
    
    # 1. 直接读取本地磁盘持久化的 ChromaDB（速度极快，几毫秒完成）
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 从本地 PDF 内存构建 BM25 (或者直接用文档切片)
    # （注：BM25 需要文档内存对象，如果文档不多，加载内存极快）
    loader = PyPDFDirectoryLoader(PDF_DIR)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # 3. 返回融合检索器
    return CustomHybridRetriever(bm25_retriever, chroma_retriever)



if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    if retriever:
        test_query = "Heart failure diagnosis and treatment"
        query_medical_db(retriever, test_query)