from langchain.prompts import PromptTemplate

# Insurance-specific template for precise policy information
insurance_policy_template = PromptTemplate(
    input_variables=["context", "question", "policy_type"],
    template=(
        """
You are an expert insurance policy analyst specializing in {policy_type} policies.

CRITICAL INSTRUCTIONS:
- Extract EXACT information from the policy document
- Include specific numbers, percentages, time periods, and amounts
- For waiting periods, grace periods, and limits: provide precise durations/amounts
- For coverage conditions: list ALL requirements and exceptions
- For definitions: provide complete policy definitions
- If information involves conditions or exceptions, include ALL relevant details
- Use the exact terminology from the policy document

POLICY CONTEXT:
{context}

QUESTION: {question}

RESPONSE REQUIREMENTS:
✅ Provide precise numerical values (days, months, percentages, amounts)
✅ Include all conditions and eligibility criteria
✅ Mention exceptions or limitations where applicable
✅ Use exact policy language for definitions
✅ If multiple scenarios exist, address all of them

ANSWER FORMAT:
Provide a direct, comprehensive answer that includes:
- Main answer with specific details
- Any conditions or requirements
- Relevant exceptions or limitations
- Exact time periods, amounts, or percentages

Answer:
"""
    ),
)

# Template for handling complex policy conditions
policy_conditions_template = PromptTemplate(
    input_variables=["context", "question", "policy_section"],
    template=(
        """
Analyze this insurance policy question focusing on conditions, eligibility, and requirements.

POLICY SECTION: {policy_section}
CONTEXT: {context}
QUESTION: {question}

ANALYSIS FRAMEWORK:
1. COVERAGE: What is covered?
2. CONDITIONS: What are the eligibility requirements?
3. LIMITATIONS: What are the restrictions or caps?
4. EXCEPTIONS: What is NOT covered or when coverage doesn't apply?
5. TIME FACTORS: Waiting periods, grace periods, policy terms?

Provide a structured answer that addresses:
- Primary coverage/benefit
- All conditions that must be met
- Any limitations or caps
- Specific time requirements
- Exceptions or exclusions

Answer:
"""
    ),
)

# Template for extracting structured policy data
policy_data_extraction_template = PromptTemplate(
    input_variables=["context", "question", "data_type"],
    template=(
        """
Extract specific {data_type} information from this insurance policy.

CONTEXT: {context}
QUESTION: {question}

EXTRACTION FOCUS:
- Numerical values (percentages, amounts, durations)
- Specific limits and caps
- Eligibility criteria
- Time periods and deadlines
- Coverage amounts and sub-limits

RESPONSE FORMAT:
Provide the exact information requested with:
- Precise numbers/percentages
- Complete conditions
- Relevant policy section references
- Any applicable exceptions

If the information has multiple parts or conditions, structure the answer clearly.

Answer:
"""
    ),
)

# Template for policy definitions and terminology
policy_definitions_template = PromptTemplate(
    input_variables=["context", "question", "term"],
    template=(
        """
Provide the exact policy definition for the requested term or concept.

CONTEXT: {context}
QUESTION: {question}
TERM TO DEFINE: {term}

REQUIREMENTS:
- Use the EXACT definition from the policy document
- Include all criteria and requirements mentioned
- Provide complete definition, not abbreviated version
- Include any specific measurements, qualifications, or conditions
- If the definition has multiple parts, include all components

Answer with the complete policy definition:
"""
    ),
)

# Template for coverage scenarios and eligibility
coverage_scenario_template = PromptTemplate(
    input_variables=["context", "question", "scenario_type"],
    template=(
        """
Analyze this {scenario_type} coverage scenario from the insurance policy.

CONTEXT: {context}
QUESTION: {question}

SCENARIO ANALYSIS:
1. IS IT COVERED? (Yes/No with explanation)
2. ELIGIBILITY REQUIREMENTS: What conditions must be met?
3. COVERAGE LIMITS: Any caps, sub-limits, or restrictions?
4. WAITING PERIODS: Any time requirements before coverage applies?
5. CLAIM PROCESS: Any specific procedures or documentation needed?
6. EXCLUSIONS: What circumstances would void coverage?

Provide a comprehensive answer that addresses the coverage question with all relevant conditions and limitations.

Answer:
"""
    ),
)
