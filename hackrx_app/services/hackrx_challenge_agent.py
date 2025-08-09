import asyncio
import json
import re
import time
from typing import Dict, Optional, List, Any

import aiohttp
import pdfplumber

from urllib.parse import urlparse

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from hackrx_app.services.document_io import download_and_hash_document
from langchain_openai import ChatOpenAI
from hackrx_app.utils.constants import GPT_PRIMARY


class HackRXChallengeAgent:
    """
    Intelligent agent that solves HackRX challenges by analyzing PDFs and executing API calls.
    No hardcoded data - everything is extracted dynamically from the PDF.
    """
    
    def __init__(self, app_logger):
        self.app_logger = app_logger
        self.llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        self.extracted_data = {}
        self.execution_memory = {}
        
    async def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract full text content from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content = ""
                for page in pdf.pages:
                    if page.extract_text():
                        content += page.extract_text() + "\n\n"
            self.app_logger.info(f"📄 Extracted {len(content)} characters from PDF")
            return content
        except Exception as e:
            self.app_logger.error(f"❌ Failed to extract PDF content: {e}")
            raise RuntimeError(f"PDF extraction failed: {str(e)}")

    async def extract_structured_data_from_pdf(self, pdf_content: str) -> Dict[str, Any]:
        """
        Use LLM to extract all mappings, endpoints, and structured data from PDF.
        This replaces all hardcoded dictionaries.
        """
        extraction_prompt = f"""
You are a data extraction specialist. Analyze this PDF content and extract ALL structured data, mappings, and relationships.

PDF CONTENT:
{pdf_content}

EXTRACTION REQUIREMENTS:
1. Find all city-landmark mappings (e.g., "Delhi -> Gateway of India")
2. Find all landmark-endpoint mappings or rules
3. Extract API endpoints, URLs, and service references
4. Identify lookup tables, reference data, key-value pairs
5. Extract any rules, conditions, or logical relationships
6. Find any flight numbers, codes, or identifiers

OUTPUT FORMAT (STRICT JSON):
{{
    "city_landmark_mappings": {{
        "City Name": "Landmark Name"
    }},
    "landmark_rules": {{
        "Landmark Name": "associated_rule_or_endpoint"
    }},
    "api_endpoints": [
        "https://full.url.here/endpoint"
    ],
    "endpoint_patterns": {{
        "pattern_name": "endpoint_suffix"
    }},
    "rules_and_logic": [
        "Rule description"
    ],
    "identifiers": {{
        "type": "value"
    }},
    "additional_mappings": {{
        "key": "value"
    }}
}}

CRITICAL: Return ONLY valid JSON. Extract EVERYTHING you can find. Be thorough.
"""
        
        try:
            self.app_logger.info("🔍 Extracting structured data from PDF using LLM...")
            result = await self.llm.ainvoke(extraction_prompt)
            response_text = result.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            extracted_data = json.loads(response_text)
            self.extracted_data = extracted_data
            
            self.app_logger.info(f"✅ Extracted {len(extracted_data.get('city_landmark_mappings', {}))} city-landmark mappings")
            self.app_logger.info(f"✅ Extracted {len(extracted_data.get('api_endpoints', []))} API endpoints")
            self.app_logger.info(f"✅ Extracted {len(extracted_data.get('rules_and_logic', []))} rules")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            self.app_logger.error(f"❌ Failed to parse LLM JSON response: {e}")
            raise RuntimeError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            self.app_logger.error(f"❌ PDF data extraction failed: {e}")
            raise RuntimeError(f"Could not extract data from PDF: {str(e)}")

    async def analyze_challenge_strategy(self, pdf_content: str) -> Dict[str, Any]:
        """
        Analyze the PDF to understand the challenge and create execution strategy.
        """
        strategy_prompt = f"""
You are a challenge analysis expert. Analyze this PDF and create a step-by-step execution strategy.

PDF CONTENT:
{pdf_content}

ANALYSIS REQUIREMENTS:
1. Identify the main objective/goal of the challenge
2. Determine what data needs to be fetched from APIs
3. Identify the sequence of operations required
4. Map out dependencies between steps
5. Identify what lookups are needed

OUTPUT FORMAT (STRICT JSON):
{{
    "challenge_objective": "Clear description of what needs to be accomplished",
    "required_data": [
        "Data item 1",
        "Data item 2"
    ],
    "execution_steps": [
        {{
            "step_number": 1,
            "action": "API_CALL|LOOKUP|PROCESS",
            "description": "What to do",
            "depends_on": ["previous_step_results"],
            "expected_output": "What this step should produce"
        }}
    ],
    "success_criteria": "How to know when challenge is complete"
}}

CRITICAL: Return ONLY valid JSON. Be specific and actionable.
"""
        
        try:
            self.app_logger.info("🎯 Analyzing challenge strategy...")
            result = await self.llm.ainvoke(strategy_prompt)
            response_text = result.content.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            strategy = json.loads(response_text)
            self.app_logger.info(f"✅ Strategy created with {len(strategy.get('execution_steps', []))} steps")
            
            return strategy
            
        except json.JSONDecodeError as e:
            self.app_logger.error(f"❌ Failed to parse strategy JSON: {e}")
            raise RuntimeError(f"Strategy analysis returned invalid JSON: {str(e)}")
        except Exception as e:
            self.app_logger.error(f"❌ Challenge strategy analysis failed: {e}")
            raise RuntimeError(f"Could not analyze challenge strategy: {str(e)}")

    async def execute_api_call(self, method: str, url: str, params: Dict = None) -> Dict[str, Any]:
        """Execute HTTP API call and return structured response"""
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, params=params) as resp:
                        status = resp.status
                        content_type = resp.headers.get("content-type", "").lower()
                        raw_text = await resp.text()
                elif method.upper() == "POST":
                    async with session.post(url, json=params) as resp:
                        status = resp.status
                        content_type = resp.headers.get("content-type", "").lower()
                        raw_text = await resp.text()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Parse response
                data = None
                if "application/json" in content_type or raw_text.strip().startswith(("{", "[")):
                    try:
                        data = json.loads(raw_text)
                    except:
                        data = raw_text
                else:
                    data = raw_text.strip()
                
                result = {
                    "status": status,
                    "data": data,
                    "raw_text": raw_text,
                    "success": 200 <= status < 300
                }
                
                self.app_logger.info(f"🌐 API {method} {url} -> {status}")
                return result
                
        except Exception as e:
            self.app_logger.error(f"❌ API call failed: {e}")
            return {
                "status": 0,
                "data": None,
                "raw_text": str(e),
                "success": False,
                "error": str(e)
            }

    async def intelligent_lookup(self, query: str, context: str = "") -> Any:
        """
        Use LLM to perform intelligent lookups in extracted data and execution memory.
        """
        lookup_prompt = f"""
You are a data lookup specialist. Find the requested information using available data.

QUERY: {query}
CONTEXT: {context}

AVAILABLE DATA:
Extracted Data: {json.dumps(self.extracted_data, indent=2)}
Execution Memory: {json.dumps(self.execution_memory, indent=2)}

LOOKUP REQUIREMENTS:
1. Search through all available data for information related to the query
2. Use fuzzy matching for city names, landmarks, etc.
3. Apply any rules or mappings found in the data
4. Return the most relevant result

OUTPUT FORMAT:
Return ONLY the found value/result. If not found, return "NOT_FOUND".
If multiple matches, return the best match.

Examples:
- Query: "landmark for Delhi" -> "Gateway of India"
- Query: "endpoint for Gateway of India" -> "getFirstCityFlightNumber"
- Query: "flight number from response" -> "AI-2024"
"""
        
        try:
            result = await self.llm.ainvoke(lookup_prompt)
            lookup_result = result.content.strip().strip('"\'')
            
            self.app_logger.info(f"🔍 Lookup '{query}' -> '{lookup_result}'")
            
            if lookup_result == "NOT_FOUND":
                return None
            return lookup_result
            
        except Exception as e:
            self.app_logger.error(f"❌ Intelligent lookup failed: {e}")
            return None

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step in the challenge strategy.
        """
        step_num = step.get("step_number", 0)
        action = step.get("action", "")
        description = step.get("description", "")
        
        self.app_logger.info(f"🔧 Executing Step {step_num}: {description}")
        
        try:
            if action == "API_CALL":
                # Use LLM to determine API call details
                api_prompt = f"""
Based on the step description and available data, determine the API call details.

STEP: {description}
AVAILABLE DATA: {json.dumps(self.extracted_data, indent=2)}
EXECUTION MEMORY: {json.dumps(self.execution_memory, indent=2)}

Determine:
1. HTTP method (GET/POST)
2. Full URL
3. Any parameters needed

OUTPUT FORMAT (JSON):
{{
    "method": "GET",
    "url": "https://full.url.here",
    "params": {{"key": "value"}}
}}
"""
                
                api_result = await self.llm.ainvoke(api_prompt)
                api_text = api_result.content.strip()
                
                if api_text.startswith("```json"):
                    api_text = api_text.replace("```json", "").replace("```", "").strip()
                elif api_text.startswith("```"):
                    api_text = api_text.replace("```", "").strip()
                
                api_details = json.loads(api_text)
                
                # Execute the API call
                response = await self.execute_api_call(
                    api_details.get("method", "GET"),
                    api_details["url"],
                    api_details.get("params")
                )
                
                # Store result in memory
                self.execution_memory[f"step_{step_num}_response"] = response
                
                return {
                    "step": step_num,
                    "action": action,
                    "result": response["data"],
                    "success": response["success"]
                }
                
            elif action == "LOOKUP":
                # Perform intelligent lookup
                lookup_result = await self.intelligent_lookup(description)
                
                # Store result in memory
                self.execution_memory[f"step_{step_num}_lookup"] = lookup_result
                
                return {
                    "step": step_num,
                    "action": action,
                    "result": lookup_result,
                    "success": lookup_result is not None
                }
                
            elif action == "PROCESS":
                # Use LLM to process data
                process_prompt = f"""
Process the data as described.

TASK: {description}
AVAILABLE DATA: {json.dumps(self.extracted_data, indent=2)}
EXECUTION MEMORY: {json.dumps(self.execution_memory, indent=2)}

Perform the processing and return the result.
"""
                
                process_result = await self.llm.ainvoke(process_prompt)
                processed_data = process_result.content.strip()
                
                # Store result in memory
                self.execution_memory[f"step_{step_num}_processed"] = processed_data
                
                return {
                    "step": step_num,
                    "action": action,
                    "result": processed_data,
                    "success": True
                }
                
        except Exception as e:
            self.app_logger.error(f"❌ Step {step_num} execution failed: {e}")
            return {
                "step": step_num,
                "action": action,
                "result": None,
                "success": False,
                "error": str(e)
            }

    async def solve_challenge(self, doc_url: str) -> str:
        """
        Main method to solve the HackRX challenge using pure agent-based approach.
        No hardcoded data - everything extracted from PDF.
        """
        try:
            # 1. Download PDF
            parsed = urlparse(doc_url)
            path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
            ext = path_ext if path_ext else "pdf"
            file_path, _doc_hash = await download_and_hash_document(doc_url, ext, self.app_logger)
            
            # 2. Extract PDF content
            pdf_content = await self.extract_pdf_content(file_path)
            
            # 3. Extract structured data from PDF (replaces hardcoded mappings)
            self.extracted_data = await self.extract_structured_data_from_pdf(pdf_content)
            
            # 4. Analyze challenge strategy
            strategy = await self.analyze_challenge_strategy(pdf_content)
            
            self.app_logger.info(f"🎯 Challenge Objective: {strategy['challenge_objective']}")
            
            # 5. Execute strategy steps
            final_result = None
            for step in strategy["execution_steps"]:
                step_result = await self.execute_step(step)
                
                if not step_result["success"]:
                    self.app_logger.error(f"❌ Step {step['step_number']} failed")
                    continue
                
                # Check if this is the final result
                if step.get("expected_output") and "final" in step.get("expected_output", "").lower():
                    final_result = step_result["result"]
                
                # Small delay between steps
                await asyncio.sleep(0.5)
            
            # 6. Determine final answer
            if not final_result:
                # Use LLM to determine final answer from execution memory
                final_prompt = f"""
Based on the execution results, determine the final answer to the challenge.

CHALLENGE OBJECTIVE: {strategy['challenge_objective']}
EXECUTION MEMORY: {json.dumps(self.execution_memory, indent=2)}

What is the final answer/result?
Return ONLY the final answer value.
"""
                
                final_llm_result = await self.llm.ainvoke(final_prompt)
                final_result = final_llm_result.content.strip()
            
            self.app_logger.info(f"✅ Challenge solved: {final_result}")
            return str(final_result)
            
        except Exception as e:
            self.app_logger.error(f"❌ Challenge solving failed: {e}")
            raise RuntimeError(f"Agent failed to solve challenge: {str(e)}")


# Main handler function that uses the agent
async def handle_hackrx_challenge(doc_url: str, app_logger, executor) -> str:
    """
    Handle HackRX challenge using the intelligent agent approach.
    No hardcoded data - everything is extracted dynamically from PDF.
    """
    agent = HackRXChallengeAgent(app_logger)
    return await agent.solve_challenge(doc_url)
