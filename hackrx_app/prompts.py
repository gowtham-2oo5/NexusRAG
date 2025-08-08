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


