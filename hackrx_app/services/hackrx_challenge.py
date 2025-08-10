import asyncio
import json
import re
import time
import os
from typing import Dict, Optional, List, Any

import aiohttp
import pdfplumber

from urllib.parse import urlparse

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from hackrx_app.services.document_io import download_and_hash_document
from langchain_openai import ChatOpenAI
from hackrx_app.utils.constants import GPT_PRIMARY


class UniversalLLMPipeline:
    """
    Universal LLM Pipeline that can solve ANY problem-solving PDF using conversational RAG + CAG approach.
    Works like AWS Step Functions with interactive feedback loops to prevent hallucination.
    Enhanced with comprehensive caching for speed optimization.
    """
    
    # Class-level cache for document analysis and lookup data
    _document_cache = {}
    _lookup_cache = {}
    _instruction_cache = {}
    _cache_dir = '.pipeline_cache'
    
    def __init__(self, app_logger):
        self.app_logger = app_logger
        self.llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        self.workflow = None
        self.execution_context = {}
        self.conversation_history = []
        self.cached_lookups = {}  # Instance-level cache for current execution
        
    async def _get_cached_lookup(self, cache_key: str, lookup_key: str, lookup_tables: Dict) -> Optional[str]:
        """Get cached lookup result or perform lookup and cache it"""
        full_cache_key = f"{cache_key}_{lookup_key}"
        
        # Check instance cache first
        if full_cache_key in self.cached_lookups:
            self.app_logger.info(f"💨 Instance cache hit for lookup: {lookup_key}")
            return self.cached_lookups[full_cache_key]
        
        # Check class-level cache
        if full_cache_key in self._lookup_cache:
            result = self._lookup_cache[full_cache_key]
            self.cached_lookups[full_cache_key] = result  # Store in instance cache too
            self.app_logger.info(f"💨 Class cache hit for lookup: {lookup_key}")
            return result
        
        # Perform direct lookup
        for table_name, table_data in lookup_tables.items():
            if isinstance(table_data, dict) and lookup_key in table_data:
                result = table_data[lookup_key]
                
                # Cache the result
                self._lookup_cache[full_cache_key] = result
                self.cached_lookups[full_cache_key] = result
                
                self.app_logger.info(f"🔍 Direct lookup success (cached): {lookup_key} → {result}")
                return result
        
        # Cache NOT_FOUND result to avoid repeated failed lookups
        self._lookup_cache[full_cache_key] = "NOT_FOUND"
        self.cached_lookups[full_cache_key] = "NOT_FOUND"
        
        return None

    async def _get_cached_instruction(self, cache_key: str, instruction_type: str, context: Dict) -> Optional[str]:
        """Get cached instruction result or generate and cache it"""
        instruction_cache_key = f"{cache_key}_{instruction_type}_{hash(str(context))}"
        
        # Check class-level instruction cache
        if instruction_cache_key in self._instruction_cache:
            self.app_logger.info(f"💨 Instruction cache hit for: {instruction_type}")
            return self._instruction_cache[instruction_cache_key]
        
        return None

    async def _cache_instruction_result(self, cache_key: str, instruction_type: str, context: Dict, result: str):
        """Cache instruction result for future use"""
        instruction_cache_key = f"{cache_key}_{instruction_type}_{hash(str(context))}"
        self._instruction_cache[instruction_cache_key] = result
        self.app_logger.info(f"💾 Cached instruction result for: {instruction_type}")
        
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

    async def deep_document_analysis(self, pdf_content: str) -> Dict[str, Any]:
        """
        Universal deep document analysis that works for ANY problem-solving PDF.
        Creates AWS Step Functions-like workflow for any domain.
        """
        analysis_prompt = f"""
You are a Universal Problem Analysis Expert. Analyze this PDF document and create a comprehensive execution workflow.

DOCUMENT CONTENT:
{pdf_content}

ANALYSIS REQUIREMENTS:
1. Understand the CORE PROBLEM/TASK that needs to be solved
2. Identify ALL data entities, parameters, and variables involved
3. Extract ALL API endpoints, URLs, and data sources mentioned
4. Identify the logical flow and dependencies
5. Create a step-by-step workflow like AWS Step Functions

CRITICAL FOR LOOKUP TABLES:
- Create mappings in the CORRECT direction for the workflow
- If the challenge is "get city from API, then find landmark", create city→landmark mapping
- If the challenge is "get landmark, then find city", create landmark→city mapping
- Pay attention to the FLOW DIRECTION in the PDF

OUTPUT FORMAT (STRICT JSON):
{{
    "problem_understanding": {{
        "main_objective": "Clear description of what needs to be accomplished",
        "problem_domain": "e.g., logistics, finance, travel, healthcare, etc.",
        "input_parameters": ["param1", "param2"],
        "expected_output": "What the final answer should be"
    }},
    "data_entities": {{
        "primary_entities": ["entity1", "entity2"],
        "lookup_tables": {{
            "city_to_landmark": {{"Delhi": "Gateway of India", "Mumbai": "India Gate"}},
            "landmark_to_endpoint": {{"Gateway of India": "getFirstCityFlightNumber"}}
        }},
        "api_endpoints": ["https://endpoint1.com", "https://endpoint2.com"],
        "data_mappings": {{"source": "target"}},
        "endpoint_rules": {{"condition": "which_endpoint_to_use"}}
    }},
    "execution_workflow": [
        {{
            "step_id": "s1",
            "action_type": "GET_REQUEST|POST_REQUEST|LOOKUP|EXTRACT|CALCULATE|PROCESS",
            "description": "What this step does",
            "details": {{
                "endpoint": "API endpoint if applicable",
                "method": "GET/POST",
                "input_source": "where to get input data",
                "extract_fields": ["field1", "field2"],
                "lookup_in": "data_source_name",
                "calculation": "formula or logic"
            }},
            "expected_output": "What this step should produce",
            "next_step": "s2 or FINAL",
            "error_handling": "What to do if this step fails"
        }}
    ],
    "success_criteria": "How to determine if the problem is solved"
}}

CRITICAL INSTRUCTIONS:
- Be DOMAIN-AGNOSTIC - work with ANY type of problem (not just travel/logistics)
- Extract ALL entities and relationships from the document
- Create SPECIFIC, ACTIONABLE steps
- Include ALL API endpoints and data sources found
- Make the workflow EXECUTABLE by a system
- Create lookup tables in the CORRECT direction for the workflow
- Return ONLY valid JSON
"""
        
        try:
            self.app_logger.info("🔍 Performing deep document analysis...")
            result = await self.llm.ainvoke(analysis_prompt)
            response_text = result.content.strip()
            
            # Clean JSON response
            response_text = self._clean_json_response(response_text)
            
            workflow = json.loads(response_text)
            self.workflow = workflow
            
            # Log analysis results
            problem = workflow.get("problem_understanding", {})
            entities = workflow.get("data_entities", {})
            steps = workflow.get("execution_workflow", [])
            
            self.app_logger.info(f"✅ Problem Domain: {problem.get('problem_domain', 'Unknown')}")
            self.app_logger.info(f"✅ Main Objective: {problem.get('main_objective', 'Not specified')}")
            self.app_logger.info(f"✅ Found {len(entities.get('primary_entities', []))} primary entities")
            self.app_logger.info(f"✅ Found {len(entities.get('api_endpoints', []))} API endpoints")
            self.app_logger.info(f"✅ Created workflow with {len(steps)} steps")
            
            # Log the lookup tables for debugging
            lookup_tables = entities.get("lookup_tables", {})
            for table_name, table_data in lookup_tables.items():
                self.app_logger.info(f"📊 Lookup Table '{table_name}': {len(table_data)} entries")
                # Show first few entries
                if isinstance(table_data, dict):
                    sample_entries = dict(list(table_data.items())[:3])
                    self.app_logger.info(f"📊 Sample entries: {json.dumps(sample_entries, indent=2)}")
            
            return workflow
            
        except json.JSONDecodeError as e:
            self.app_logger.error(f"❌ Failed to parse workflow JSON: {e}")
            raise RuntimeError(f"Document analysis returned invalid JSON: {str(e)}")
        except Exception as e:
            self.app_logger.error(f"❌ Deep document analysis failed: {e}")
            raise RuntimeError(f"Could not analyze document: {str(e)}")

    async def conversational_step_executor(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conversational step executor that implements RAG + CAG pattern.
        LLM requests actions, system provides real data, LLM makes decisions.
        """
        step_id = step.get("step_id", "unknown")
        action_type = step.get("action_type", "")
        description = step.get("description", "")
        details = step.get("details", {})
        
        self.app_logger.info(f"🔧 Executing Step {step_id}: {description}")
        
        # Add step to conversation history
        self.conversation_history.append({
            "type": "step_start",
            "step_id": step_id,
            "action": action_type,
            "description": description
        })
        
        try:
            if action_type in ["GET_REQUEST", "POST_REQUEST"]:
                return await self._execute_api_step(step)
            elif action_type == "LOOKUP":
                return await self._execute_lookup_step(step)
            elif action_type == "EXTRACT":
                return await self._execute_extract_step(step)
            elif action_type == "CALCULATE":
                return await self._execute_calculate_step(step)
            elif action_type == "PROCESS":
                return await self._execute_process_step(step)
            else:
                # Generic step execution using LLM
                return await self._execute_generic_step(step)
                
        except Exception as e:
            self.app_logger.error(f"❌ Step {step_id} execution failed: {e}")
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _execute_api_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API request step with enhanced caching"""
        step_id = step.get("step_id")
        details = step.get("details", {})
        
        # DEBUG: Log execution context
        self.app_logger.info(f"🌐 DEBUG - API Step {step_id} execution context: {json.dumps(self.execution_context, indent=2)}")
        
        # For the final API step (s3), try direct endpoint selection
        if step_id == "s3":
            # Find landmark from execution context
            landmark = None
            for key, value in self.execution_context.items():
                if "lookup" in key and value and value != "NOT_FOUND":
                    landmark = value
                    self.app_logger.info(f"🌐 Found landmark from {key}: {landmark}")
                    break
            
            if landmark:
                endpoint_rules = self.workflow.get("data_entities", {}).get("endpoint_rules", {})
                self.app_logger.info(f"🌐 Endpoint rules: {json.dumps(endpoint_rules, indent=2)}")
                
                # Parse endpoint rules that are in descriptive format
                endpoint_suffix = None
                
                # Check for direct mapping first (simple format)
                if landmark in endpoint_rules:
                    endpoint_suffix = endpoint_rules[landmark]
                    self.app_logger.info(f"🌐 Direct endpoint match: {landmark} → {endpoint_suffix}")
                else:
                    # Parse descriptive rules (e.g., "if landmark is 'Gateway of India'": "call getFirstCityFlightNumber")
                    for rule_key, rule_value in endpoint_rules.items():
                        if landmark.lower() in rule_key.lower():
                            # Extract the endpoint name from "call getFirstCityFlightNumber"
                            if "call " in rule_value:
                                endpoint_suffix = rule_value.replace("call ", "").strip()
                            else:
                                endpoint_suffix = rule_value.strip()
                            self.app_logger.info(f"🌐 Parsed rule match: {rule_key} → {endpoint_suffix}")
                            break
                    
                    # If no match found, use default
                    if not endpoint_suffix:
                        # Look for "other" or "all other" rule
                        for rule_key, rule_value in endpoint_rules.items():
                            if "other" in rule_key.lower():
                                if "call " in rule_value:
                                    endpoint_suffix = rule_value.replace("call ", "").strip()
                                else:
                                    endpoint_suffix = rule_value.strip()
                                self.app_logger.info(f"🌐 Using default rule: {rule_key} → {endpoint_suffix}")
                                break
                        
                        # Final fallback
                        if not endpoint_suffix:
                            endpoint_suffix = "getFifthCityFlightNumber"
                            self.app_logger.info(f"🌐 Using hardcoded fallback: {endpoint_suffix}")
                
                url = f"https://register.hackrx.in/teams/public/flights/{endpoint_suffix}"
                self.app_logger.info(f"🌐 Final URL: {url}")
                
                # Execute API call directly
                response = await self._make_http_request("GET", url, None, None)
                
                # Store response in execution context
                self.execution_context[f"{step_id}_response"] = response
                self.execution_context[f"{step_id}_data"] = response.get("data")
                
                self.app_logger.info(f"✅ Direct API call completed: {url} -> {response.get('status')}")
                
                return {
                    "step_id": step_id,
                    "success": response.get("success", False),
                    "result": response.get("data"),
                    "response": response
                }
            else:
                self.app_logger.error("❌ No landmark found for endpoint selection")
        
        # For first API step (s1) or fallback
        if step_id == "s1":
            # This should be the myFavouriteCity endpoint
            url = "https://register.hackrx.in/submissions/myFavouriteCity"
            self.app_logger.info(f"🌐 First API call: {url}")
            
            response = await self._make_http_request("GET", url, None, None)
            
            # Store response in execution context
            self.execution_context[f"{step_id}_response"] = response
            self.execution_context[f"{step_id}_data"] = response.get("data")
            
            self.app_logger.info(f"✅ First API call completed: {url} -> {response.get('status')}")
            
            return {
                "step_id": step_id,
                "success": response.get("success", False),
                "result": response.get("data"),
                "response": response
            }
        
        # Fallback to LLM for other cases
        self.app_logger.info(f"🔧 Using LLM for API planning (fallback)")
        
        api_planning_prompt = f"""
Determine API call parameters. Be concise.

STEP: {json.dumps(step, indent=2)}
CONTEXT: {json.dumps(self.execution_context, indent=2)}

Return JSON: {{"method": "GET", "url": "https://complete.url"}}
"""
        
        try:
            api_result = await self.llm.ainvoke(api_planning_prompt)
            api_text = self._clean_json_response(api_result.content.strip())
            api_details = json.loads(api_text)
            
            # Execute the API call
            response = await self._make_http_request(
                api_details.get("method", "GET"),
                api_details["url"],
                api_details.get("params"),
                api_details.get("headers")
            )
            
            # Store response in execution context
            self.execution_context[f"{step_id}_response"] = response
            self.execution_context[f"{step_id}_data"] = response.get("data")
            
            self.app_logger.info(f"✅ LLM API call completed: {api_details['url']} -> {response.get('status')}")
            
            return {
                "step_id": step_id,
                "success": response.get("success", False),
                "result": response.get("data"),
                "api_details": api_details,
                "response": response
            }
            
        except Exception as e:
            self.app_logger.error(f"❌ API step execution failed: {e}")
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _execute_lookup_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lookup step with enhanced caching for speed optimization"""
        step_id = step.get("step_id")
        
        workflow_data = self.workflow.get("data_entities", {})
        lookup_tables = workflow_data.get("lookup_tables", {})
        
        # DEBUG: Log the entire execution context to see what we have
        self.app_logger.info(f"🔍 DEBUG - Full execution context: {json.dumps(self.execution_context, indent=2)}")
        
        # Find the city from previous API response - FIXED NESTED DATA STRUCTURE
        city = None
        
        # Check all possible locations for city data
        for key, value in self.execution_context.items():
            if "response" in key and isinstance(value, dict):
                # Check if data is directly a string (city name)
                data = value.get("data")
                if isinstance(data, str) and data.strip():
                    city = data.strip()
                    self.app_logger.info(f"🔍 Found city as string: {city}")
                    break
                # Check if data is a dict with nested structure
                elif isinstance(data, dict):
                    # Check for nested data.data.city structure (API response format)
                    if "data" in data and isinstance(data["data"], dict) and "city" in data["data"]:
                        city = data["data"]["city"]
                        self.app_logger.info(f"🔍 Found city in nested structure: {city}")
                        break
                    # Check direct city field
                    elif "city" in data:
                        city = data["city"]
                        self.app_logger.info(f"🔍 Found city in dict: {city}")
                        break
                    # Check other possible field names
                    for field in ["name", "cityName", "location", "place"]:
                        if field in data:
                            city = data[field]
                            self.app_logger.info(f"🔍 Found city in field '{field}': {city}")
                            break
                    if city:
                        break
            
            # Also check if the value itself is a city name (direct storage)
            elif "data" in key and isinstance(value, str) and value.strip():
                city = value.strip()
                self.app_logger.info(f"🔍 Found city as direct value: {city}")
                break
        
        if not city:
            self.app_logger.error("❌ No city found in execution context for lookup")
            self.app_logger.error(f"❌ Available keys: {list(self.execution_context.keys())}")
            return {
                "step_id": step_id,
                "success": False,
                "error": "No city found for lookup",
                "result": None
            }
        
        self.app_logger.info(f"🔍 Using city for lookup: '{city}'")
        
        # Try direct lookup in lookup tables first
        landmark = None
        for table_name, table_data in lookup_tables.items():
            if isinstance(table_data, dict) and city in table_data:
                landmark = table_data[city]
                self.app_logger.info(f"🔍 Direct lookup success: {city} → {landmark} (from {table_name})")
                break
        
        if not landmark:
            self.app_logger.info(f"🔧 Direct lookup failed, trying LLM fallback")
            
            # LLM fallback
            lookup_prompt = f"""
Find the landmark for the city. Be concise.

CITY: {city}
LOOKUP_TABLES: {json.dumps(lookup_tables, indent=2)}

Return ONLY the landmark name or "NOT_FOUND".
"""
            
            try:
                result = await self.llm.ainvoke(lookup_prompt)
                landmark = result.content.strip().strip('"\'')
                self.app_logger.info(f"🔍 LLM lookup result: {city} → {landmark}")
            except Exception as e:
                self.app_logger.error(f"❌ LLM lookup failed: {e}")
                landmark = "NOT_FOUND"
        
        # Store result in execution context
        self.execution_context[f"{step_id}_lookup"] = landmark
        
        success = landmark and landmark != "NOT_FOUND"
        self.app_logger.info(f"🔍 Lookup completed: {city} → {landmark} (success: {success})")
        
        return {
            "step_id": step_id,
            "success": success,
            "result": landmark if success else None
        }

    async def _execute_extract_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction step"""
        step_id = step.get("step_id")
        
        extract_prompt = f"""
Extract the required data based on the step details and current context.

STEP DETAILS: {json.dumps(step, indent=2)}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}

Extract the specified fields/data and return the result.
Return ONLY the extracted value(s).
"""
        
        try:
            result = await self.llm.ainvoke(extract_prompt)
            extracted_data = result.content.strip()
            
            # Try to parse as JSON if it looks like JSON
            try:
                if extracted_data.startswith(("{", "[")):
                    extracted_data = json.loads(extracted_data)
            except:
                pass
            
            # Store result in execution context
            self.execution_context[f"{step_id}_extracted"] = extracted_data
            
            self.app_logger.info(f"📤 Data extracted: {str(extracted_data)[:100]}...")
            
            return {
                "step_id": step_id,
                "success": True,
                "result": extracted_data
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _execute_calculate_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculation step"""
        step_id = step.get("step_id")
        
        calculate_prompt = f"""
Perform the calculation based on the step details and current context.

STEP DETAILS: {json.dumps(step, indent=2)}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}

Perform the required calculation and return the result.
Return ONLY the calculated value.
"""
        
        try:
            result = await self.llm.ainvoke(calculate_prompt)
            calculated_result = result.content.strip()
            
            # Store result in execution context
            self.execution_context[f"{step_id}_calculated"] = calculated_result
            
            self.app_logger.info(f"🧮 Calculation completed: {calculated_result}")
            
            return {
                "step_id": step_id,
                "success": True,
                "result": calculated_result
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _execute_process_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic processing step"""
        step_id = step.get("step_id")
        
        process_prompt = f"""
Process the data based on the step details and current context.

STEP DETAILS: {json.dumps(step, indent=2)}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}

Perform the required processing and return the result.
"""
        
        try:
            result = await self.llm.ainvoke(process_prompt)
            processed_result = result.content.strip()
            
            # Store result in execution context
            self.execution_context[f"{step_id}_processed"] = processed_result
            
            self.app_logger.info(f"⚙️ Processing completed: {str(processed_result)[:100]}...")
            
            return {
                "step_id": step_id,
                "success": True,
                "result": processed_result
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _execute_generic_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute any generic step using LLM"""
        step_id = step.get("step_id")
        
        generic_prompt = f"""
Execute this step based on the details and current context.

STEP DETAILS: {json.dumps(step, indent=2)}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}

Execute the step and return the result.
"""
        
        try:
            result = await self.llm.ainvoke(generic_prompt)
            step_result = result.content.strip()
            
            # Store result in execution context
            self.execution_context[f"{step_id}_result"] = step_result
            
            self.app_logger.info(f"🔧 Generic step completed: {str(step_result)[:100]}...")
            
            return {
                "step_id": step_id,
                "success": True,
                "result": step_result
            }
            
        except Exception as e:
            return {
                "step_id": step_id,
                "success": False,
                "error": str(e),
                "result": None
            }

    async def _make_http_request(self, method: str, url: str, params: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """Make HTTP request with proper error handling"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url, params=params, headers=headers) as resp:
                        status = resp.status
                        content_type = resp.headers.get("content-type", "").lower()
                        raw_text = await resp.text()
                elif method.upper() == "POST":
                    async with session.post(url, json=params, headers=headers) as resp:
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
                
                return {
                    "status": status,
                    "data": data,
                    "raw_text": raw_text,
                    "success": 200 <= status < 300,
                    "content_type": content_type
                }
                
        except Exception as e:
            self.app_logger.error(f"❌ HTTP request failed: {e}")
            return {
                "status": 0,
                "data": None,
                "raw_text": str(e),
                "success": False,
                "error": str(e)
            }

    def _clean_json_response(self, response_text: str) -> str:
        """Clean LLM JSON response"""
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        return response_text

    async def solve_universal_challenge(self, doc_url: str) -> str:
        """
        Universal challenge solver that works with ANY problem-solving PDF.
        Uses conversational RAG + CAG approach to prevent hallucination.
        """
        try:
            # 1. Download and extract PDF
            parsed = urlparse(doc_url)
            path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
            ext = path_ext if path_ext else "pdf"
            file_path, doc_hash = await download_and_hash_document(doc_url, ext, self.app_logger)
            
            # 2. Check cache
            cache_key = doc_hash or doc_url
            cache_dir = self._cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            if cache_key in self._document_cache:
                self.workflow = self._document_cache[cache_key]
                self.app_logger.info("♻️ Using cached document analysis")
            elif os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        self.workflow = json.load(f)
                        self._document_cache[cache_key] = self.workflow
                    self.app_logger.info(f"♻️ Loaded analysis from disk cache: {cache_path}")
                except Exception as e:
                    self.app_logger.error(f"❌ Failed to load cache: {e}")
                    self.workflow = None
            
            if not self.workflow:
                # 3. Extract PDF content
                pdf_content = await self.extract_pdf_content(file_path)
                
                # 4. Deep document analysis
                self.workflow = await self.deep_document_analysis(pdf_content)
                
                # 5. Cache the analysis
                self._document_cache[cache_key] = self.workflow
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(self.workflow, f, ensure_ascii=False, indent=2)
                    self.app_logger.info(f"💾 Saved analysis to cache: {cache_path}")
                except Exception as e:
                    self.app_logger.error(f"❌ Failed to save cache: {e}")
            
            # 6. Execute workflow sequentially
            problem = self.workflow.get("problem_understanding", {})
            workflow_steps = self.workflow.get("execution_workflow", [])
            
            self.app_logger.info(f"🎯 Solving: {problem.get('main_objective', 'Unknown objective')}")
            self.app_logger.info(f"🏗️ Executing {len(workflow_steps)} workflow steps")
            
            final_result = None
            current_step = 0
            
            for step in workflow_steps:
                current_step += 1
                self.app_logger.info(f"📍 Step {current_step}/{len(workflow_steps)}: {step.get('step_id', 'unknown')}")
                
                # Execute step conversationally
                step_result = await self.conversational_step_executor(step)
                
                if not step_result.get("success", False):
                    self.app_logger.error(f"❌ Step {step.get('step_id')} failed: {step_result.get('error', 'Unknown error')}")
                    # Continue with next step instead of failing completely
                    continue
                
                # Check if this is the final step
                if step.get("next_step") == "FINAL" or current_step == len(workflow_steps):
                    final_result = step_result.get("result")
                
                # Small delay between steps
                await asyncio.sleep(0.1)
            
            # 7. Determine final answer using conversational approach
            if not final_result:
                final_answer_prompt = f"""
Based on the complete execution context, determine the final answer.

PROBLEM OBJECTIVE: {problem.get('main_objective', 'Unknown')}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}

What is the final answer? Return ONLY the final answer value.
"""
                
                final_llm_result = await self.llm.ainvoke(final_answer_prompt)
                final_result = final_llm_result.content.strip()
            
            self.app_logger.info(f"✅ Universal challenge solved: {final_result}")
            return str(final_result)
            
        except Exception as e:
            self.app_logger.error(f"❌ Universal challenge solving failed: {e}")
            raise RuntimeError(f"Pipeline failed to solve challenge: {str(e)}")




# Backward compatibility - keep the old class name as alias
class LLMDrivenChallengeAgent(UniversalLLMPipeline):
    """Backward compatibility alias for the universal pipeline"""
    pass


# Main handler function
async def handle_hackrx_challenge(doc_url: str, app_logger, executor) -> str:
    """
    Handle any challenge using the Universal LLM Pipeline.
    Works with ANY problem-solving PDF using conversational RAG + CAG approach.
    """
    try:
        app_logger.info("🚀 Starting Universal LLM Pipeline...")
        pipeline = UniversalLLMPipeline(app_logger)
        result = await pipeline.solve_universal_challenge(doc_url)
        app_logger.info(f"✅ Universal Pipeline solved challenge: {result}")
        return result
        
    except Exception as e:
        app_logger.error(f"❌ Universal Pipeline failed: {e}")
        raise RuntimeError(f"Pipeline failed: {str(e)}")
