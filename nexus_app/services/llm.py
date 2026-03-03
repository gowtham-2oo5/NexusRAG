import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from openai import RateLimitError

from nexus_app.prompts import (
    domain_detection_template,
    enhanced_prompt_template,
    fact_checking_template,
    factual_content_template,

    challenge_analysis_template,
    api_execution_template
)
from nexus_app.utils.constants import GPT_FALLBACK, GPT_PRIMARY


async def detect_document_domain(docs: List[Document]) -> str:
    sample_content = "\n".join([doc.page_content for doc in docs[:3]])[:2000]
    llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.1)
    prompt = domain_detection_template.format(document_content=sample_content)
    result = await llm.ainvoke(prompt)
    return result.content.strip()


def is_general_knowledge_question(question: str) -> bool:
    general_knowledge_indicators = [
        "what is the capital of",
        "where can we find",
        "what are clouds made of",
        "how many",
        "who is",
        "what is the name of",
        "when was",
        "how old is",
        "what color is",
        "how tall is",
        "what does",
        "where is",
        "which country",
        "what planet",
        "what galaxy",
        "what ocean",
        "what continent",
    ]
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in general_knowledge_indicators)


def compress_context_with_bm25(query: str, context_docs: List[Document], max_sentences: int = 12) -> str:
    from rank_bm25 import BM25Okapi

    texts = [doc.page_content for doc in context_docs if doc and doc.page_content]
    if not texts:
        return ""

    sentences: list[str] = []
    for text in texts:
        cand = re.split(r"(?<=[.!?])\s+", text)
        for s in cand:
            s_clean = s.strip()
            if 20 <= len(s_clean) <= 600:
                sentences.append(s_clean)

    if not sentences:
        return "\n\n".join(texts)[:4000]

    tokenized_corpus = [sent.lower().split() for sent in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    seen: set[str] = set()
    compressed: list[str] = []
    for sent, _ in ranked:
        key = sent[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        compressed.append(sent)
        if len(compressed) >= max_sentences:
            break
    return "\n\n".join(compressed)


async def ask_with_context(question: str, context_docs: List[Document], domain: str, app_logger) -> str:
    # For factual content domains, use full context to avoid text corruption
    if domain.lower() in ['technology/software', 'news', 'media']:
        # Use full context without BM25 compression to preserve multi-language text integrity
        context = "\n\n".join([doc.page_content for doc in context_docs if doc and doc.page_content])[:8000]
        app_logger.info(f"🌐 Using full context for factual domain: {domain}")
    else:
        context = compress_context_with_bm25(question, context_docs)

    def extract_hex_token(text: str) -> Optional[str]:
        matches = re.findall(r"\b[a-fA-F0-9]{32,}\b", text)
        if not matches:
            return None
        return sorted(matches, key=lambda s: len(s), reverse=True)[0]

    q_lower = question.lower()
    token = extract_hex_token(context)

    if token:
        if ("secret token" in q_lower) or ("get the token" in q_lower) or ("provide the token" in q_lower):
            return token
        count_match = re.search(r"how many\s+([a-zA-Z0-9])'?s?\s+are\s+there\s+in\s+the\s+token", q_lower)
        if count_match:
            ch = count_match.group(1)
            return str(token.lower().count(ch.lower()))

    if is_general_knowledge_question(question):
        app_logger.info(f"🔍 Detected general knowledge question: {question[:50]}...")
        prompt = fact_checking_template.format(context=context, question=question, domain=domain)
    elif domain.lower() in ['technology/software', 'news', 'media']:
        app_logger.info(f"🎯 Using factual content template for domain: {domain}")
        prompt = factual_content_template.format(context=context, question=question, domain=domain)
    else:
        prompt = enhanced_prompt_template.format(context=context, question=question, domain=domain)

    try:
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.0)
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except RateLimitError:
        app_logger.warning(
            "⚠️ GPT-4o rate limited. Falling back to 4o-mini (you have 500 RPM for mini)"
        )
        fallback_llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.0)
        result = await fallback_llm.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        app_logger.exception("❌ GPT error")
        raise e


async def analyze_challenge_document(document_content: str, app_logger) -> str:
    """Analyze a challenge document and generate step-by-step instructions"""
    try:
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        prompt = challenge_analysis_template.format(document_content=document_content)
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except RateLimitError:
        app_logger.warning("⚠️ GPT-4o rate limited. Falling back to 4o-mini for challenge analysis")
        fallback_llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.1)
        result = await fallback_llm.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        app_logger.exception("❌ Challenge analysis error")
        raise e


async def execute_instruction_with_llm(instruction: str, previous_results: str, memory_data: str, app_logger) -> str:
    """Execute an instruction step using LLM guidance"""
    try:
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        prompt = api_execution_template.format(
            instruction=instruction,
            previous_results=previous_results,
            memory_data=memory_data
        )
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except RateLimitError:
        app_logger.warning("⚠️ GPT-4o rate limited. Falling back to 4o-mini for instruction execution")
        fallback_llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.1)
        result = await fallback_llm.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        app_logger.exception("❌ Instruction execution error")
        raise e
