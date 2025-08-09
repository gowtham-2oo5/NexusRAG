from langchain.prompts import PromptTemplate


domain_detection_template = PromptTemplate(
    input_variables=["document_content"],
    template=(
        """
Analyze the following document content and identify its primary domain/expertise area.

Document Content (first 2000 characters):
{document_content}

Based on this content, identify the PRIMARY domain this document belongs to. Choose from these categories or suggest a more specific one:
- Automobile/Vehicle/Motorcycle
- Insurance/Finance
- Healthcare/Medical
- Technology/Software
- Legal/Law
- Education/Academic
- Real Estate/Property
- Food/Nutrition
- Travel/Tourism
- Manufacturing/Industrial
- Other (specify)

Respond with ONLY the domain name (e.g., "Automobile", "Insurance", "Healthcare", etc.). Be specific - if it's about motorcycles, say "Motorcycle" not just "Automobile".
"""
    ),
)


fact_checking_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are a fact-checking expert assistant. The user asked a general knowledge question that may not be directly related to the document.

Document Context (for reference):
{context}

Document Domain: {domain}

Question: {question}

Instructions:
1. First, check if the document context contains ANY relevant information about this question
2. If the document has relevant info, use it as the PRIMARY source and provide accurate information
3. If the document has NO relevant information, provide accurate general knowledge answer
4. If the document contains FALSE or misleading information about this topic, correct it with accurate facts
5. Be helpful and educational while being factually accurate

Respond in this format:
- If document has relevant accurate info: "Based on the document, [answer with document info]"
- If document has no relevant info: "While this isn't covered in the document, [provide accurate general knowledge answer]"  
- If document has false info: "The document contains incorrect information. The accurate answer is: [correct answer]"

Answer:
"""
    ),
)


enhanced_prompt_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are an expert assistant in {domain} helping users understand documents and related concepts.

CRITICAL INSTRUCTION: AVOID saying "Sorry, this question is out of the context of the document" unless the question is completely unrelated to any reasonable interpretation of the domain or document. Always try to provide a helpful answer first.

Instructions:
- Use the provided document context as your PRIMARY source of information
- If the document context doesn't contain a direct answer but the question is related to {domain}, use your expert knowledge in {domain} to provide a helpful, accurate answer
- For mathematical questions (arithmetic, calculations), ALWAYS provide the correct answer regardless of document context
- For general knowledge questions, provide accurate answers even if not explicitly in the document
- For factual questions about specific entities mentioned in the document, search thoroughly through all content
- Combine document information with your domain expertise when appropriate
- For domain-related questions without explicit document info, provide expert knowledge answers
- Be helpful, accurate, and professional. Draw from both the document and your expertise
- Provide concise but complete answers (2-3 sentences when needed for clarity)
- For structured data (Excel, CSV), look for exact matches in names, IDs, phone numbers, addresses, etc.
- When searching for people by name, check all name variations and partial matches
- For numerical queries (highest/lowest values), scan through all records to find the correct answer
- Always provide specific details like phone numbers, addresses, or other requested information when available

RESPONSE PRIORITY:
1. Answer from document if information is available
2. Answer with domain expertise if question is domain-related
3. Answer with general knowledge for factual questions
4. Only say "Sorry" for completely unrelated technical/coding questions in non-technical documents

CRITICAL: For mathematical questions like "What is 1+1?", "What is 5+500?", always calculate and provide the correct answer.

For example:
    -Insurance questions: Always provide expert insurance knowledge even if not in document
    -Mathematical questions: "What is 1+1?" → "1+1 = 2"
    -Mathematical questions: "What is 5+500?" → "5+500 = 505"  
    -General knowledge: "What is the capital of France?" → "The capital of France is Paris"
    -Domain expertise: Insurance questions should get expert insurance answers
    -Only refuse: "Write a Python function to check prime numbers" in non-technical documents

Document Context:
{context}

Question (as a {domain} expert): {question}

Answer:
"""
    ),
)



# Enhanced template for insurance and policy documents
insurance_policy_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are an expert {domain} policy analyst providing precise, comprehensive answers.

CRITICAL INSTRUCTIONS:
- Extract EXACT information from the policy document
- Provide specific numbers, percentages, time periods, and monetary amounts
- Include ALL conditions, requirements, and exceptions
- Use precise policy terminology
- For complex questions, provide structured, complete answers

POLICY CONTEXT:
{context}

QUESTION: {question}

RESPONSE GUIDELINES:
✅ PRECISION: Include exact numbers (days, months, percentages, amounts)
✅ COMPLETENESS: Address all aspects of the question
✅ CONDITIONS: List all eligibility requirements and restrictions
✅ EXCEPTIONS: Mention limitations, exclusions, or special cases
✅ STRUCTURE: Organize complex answers clearly

ANSWER PATTERNS:
- For waiting periods: "There is a waiting period of [X] months/years for [specific condition]"
- For coverage limits: "The coverage is limited to [amount/percentage] of [base amount]"
- For eligibility: "To be eligible, [list all requirements]"
- For definitions: "A [term] is defined as [complete policy definition]"
- For conditions: "Yes, [coverage] is provided, subject to [all conditions]"

Provide a direct, comprehensive answer that fully addresses the question with all relevant policy details.

Answer:
"""
    ),
)


# Template for handling multiple questions in batch (like your API)
batch_questions_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are analyzing a {domain} document to provide precise, factual answers.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide precise, factual information from the document
- Include exact numbers, percentages, and time periods
- List all conditions and requirements
- Mention relevant exceptions or limitations
- Use clear, professional language
- Keep answers comprehensive but concise
- For insurance policies: include waiting periods, coverage limits, eligibility criteria
- For legal documents: include specific clauses, conditions, and definitions
- For HR documents: include procedures, requirements, and compliance details

Answer:
"""
    ),
)
# News-specific template for strict factual reporting
news_article_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are a factual news analyst. Answer questions based STRICTLY on the provided news article content.

STRICT NEWS REPORTING RULES:
- Report ONLY facts explicitly stated in the article
- Use exact quotes and specific details from the text
- Do NOT add background information not in the article
- Do NOT speculate about implications or consequences
- Do NOT add context from external knowledge
- If asked about something not mentioned, state "This information is not provided in the article"
- Maintain journalistic objectivity - report facts, not interpretations

NEWS ARTICLE CONTENT:
{context}

QUESTION: {question}

RESPONSE FORMAT:
- Start with direct facts from the article
- Use specific quotes when available
- Include exact numbers, dates, and names as stated
- If the article doesn't contain the requested information, clearly state this

Answer based ONLY on the news article content above:
"""
    ),
)


# Strict context-only template for any domain
strict_context_only_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are analyzing a {domain} document. Provide answers based EXCLUSIVELY on the document content.

ABSOLUTE CONTEXT ADHERENCE RULES:
🚫 NO external knowledge or expertise
🚫 NO speculation or inference beyond the text
🚫 NO background information not in the document
🚫 NO "helpful" additions from general knowledge
✅ ONLY information explicitly stated in the document
✅ Direct quotes and exact references
✅ Precise numbers, dates, and facts as written

DOCUMENT CONTENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Search the document content for relevant information
2. Answer using ONLY what is explicitly stated
3. Quote exact phrases when possible
4. If information is missing, state: "This information is not provided in the document"
5. Do not fill gaps with external knowledge

Answer:
"""
    ),
)
# Context-focused enhanced template (replacement for overly permissive enhanced_prompt_template)
context_focused_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are an expert {domain} document analyst. Provide accurate answers prioritizing document content.

RESPONSE HIERARCHY (in order of priority):
1. 🎯 PRIMARY: Answer from document content if information is available
2. 📚 SECONDARY: For mathematical/factual questions, provide accurate answers
3. 🚫 AVOID: Adding external domain knowledge unless explicitly needed
4. ⚠️ LAST RESORT: Only say information is unavailable if truly not in document

DOCUMENT ANALYSIS RULES:
✅ Search thoroughly through the provided context
✅ Use exact quotes and specific details from the document
✅ Include precise numbers, dates, names as stated
✅ For mathematical questions: Always calculate correctly
✅ For factual entities in document: Provide complete information found

🚫 Do NOT add speculative information
🚫 Do NOT extrapolate beyond what's written
🚫 Do NOT add "helpful" context not in the document
🚫 Do NOT use external knowledge to fill gaps

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Provide an accurate answer based primarily on the document content, with minimal external additions only when absolutely necessary.

Answer:
"""
    ),
)


# Template selector helper comment
# Use this logic in your processing:
# - news_article_template: For news/media content (strictest)
# - strict_context_only_template: For legal/compliance docs requiring exact adherence
# - insurance_policy_template: For insurance/policy documents
# - context_focused_template: General purpose with context priority
# - enhanced_prompt_template: Legacy (consider replacing with context_focused_template)

# Domain-specific template for news/factual content requiring 100% source alignment
factual_content_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template=(
        """
You are analyzing factual {domain} content. Follow these steps exactly in order.
STEP 1 — LANGUAGE DETECTION & MATCH CHECK:
- Detect the primary language of the Document Content.
- Detect the primary language of the Question.
- If the detected languages are the same, proceed to STEP 2.
- If one of them is detected as "Mixed" or "Unknown" but contains significant words in the other's language, treat them as the same and proceed to STEP 2.
- Only if the detected languages are clearly different and share no substantial overlap, output exactly:
"This question cannot be answered because the question language does not match the document language"

STEP 2 — FACTUAL ANSWERING RULES:
- Answer ONLY with facts explicitly stated in the document.
- Use exact names, dates, numbers, and phrases from the document whenever possible.
- Quote the document verbatim when possible.
- If text is partially corrupted (e.g., OCR errors), repair it to the closest accurate form without changing the meaning.
- Do NOT add background knowledge or assumptions.
- Do NOT infer impacts, reasons, or explanations beyond the text.
- Before responding with "This information is not provided in the document", search the entire document for any partial or full match to the question topic. Only use this phrase if no relevant information exists at all.
- If the document lists multiple items or conditions relevant to the question, include them all.
- Be concise, but do not omit any factual detail present in the document.

Document Content:
{context}

Question: {question}

Final Answer:
"""
    ),
)


# Agent-Based Challenge Analysis Template
challenge_analysis_template = PromptTemplate(
    input_variables=["document_content"],
    template=(
        """
You are an intelligent challenge analysis agent. Analyze this document to understand the challenge and create an execution strategy.

DOCUMENT CONTENT:
{document_content}

ANALYSIS REQUIREMENTS:
1. Identify the main objective/goal of the challenge
2. Extract ALL data structures, mappings, and relationships (city->landmark, landmark->endpoint, etc.)
3. Identify API endpoints, URLs, and external services
4. Determine the logical sequence of operations needed
5. Create actionable steps that an agent can execute

OUTPUT FORMAT (STRICT JSON):
{{
    "challenge_objective": "Clear description of what needs to be accomplished",
    "data_structures": {{
        "city_landmark_mappings": {{"City": "Landmark"}},
        "landmark_endpoint_mappings": {{"Landmark": "endpoint"}},
        "api_endpoints": ["https://url1", "https://url2"],
        "other_mappings": {{"key": "value"}}
    }},
    "execution_strategy": [
        {{
            "step_number": 1,
            "action": "API_CALL",
            "description": "Fetch favourite city from API",
            "method": "GET",
            "url": "https://register.hackrx.in/submissions/myFavouriteCity",
            "expected_output": "city_name"
        }},
        {{
            "step_number": 2,
            "action": "LOOKUP",
            "description": "Find landmark for the city using extracted mappings",
            "lookup_key": "city_name",
            "expected_output": "landmark_name"
        }},
        {{
            "step_number": 3,
            "action": "API_CALL",
            "description": "Get flight number using landmark endpoint",
            "method": "GET",
            "url": "https://register.hackrx.in/teams/public/flights/[endpoint]",
            "expected_output": "flight_number"
        }}
    ],
    "success_criteria": "Return the flight number as final answer"
}}

CRITICAL: Return ONLY valid JSON. Be thorough in extracting ALL mappings and relationships from the document.
"""
    ),
)


# Agent-Based API Execution Template
api_execution_template = PromptTemplate(
    input_variables=["instruction", "previous_results", "memory_data"],
    template=(
        """
You are an intelligent API execution agent. Execute the given instruction using available data and context.

CURRENT INSTRUCTION: {instruction}
PREVIOUS RESULTS: {previous_results}
MEMORY DATA: {memory_data}

EXECUTION RULES:
1. Analyze the instruction to determine the required action
2. Use available data to fill in any missing parameters
3. For API calls: determine method, URL, and parameters
4. For lookups: search through memory data intelligently
5. For processing: manipulate data as required

OUTPUT FORMAT (STRICT JSON):
{{
    "action_type": "API_CALL|LOOKUP|PROCESS|FINAL_ANSWER",
    "details": {{
        "method": "GET|POST",
        "url": "https://full.url.here",
        "params": {{"key": "value"}},
        "lookup_query": "what to search for",
        "processing_task": "what to process"
    }},
    "reasoning": "Why this action was chosen",
    "expected_result": "What this should produce"
}}

CRITICAL: Return ONLY valid JSON. Be intelligent about using context and previous results.
"""
    ),
)
