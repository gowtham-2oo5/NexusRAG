"""
HackRX Challenge Handler - Pure LLM-Driven Agent
The LLM analyzes the PDF and provides complete execution instructions.
The application is just a generic executor that follows LLM commands.
"""

import asyncio
import json
import re
from typing import Dict, Optional, List, Any
from urllib.parse import urlparse

import aiohttp
import pdfplumber

from hackrx_app.services.document_io import download_and_hash_document
from langchain_openai import ChatOpenAI
from hackrx_app.utils.constants import GPT_PRIMARY


class LLMDrivenChallengeAgent:
    """
    Pure LLM-driven agent. The LLM analyzes PDF and provides complete execution plan.
    Application just executes the LLM's instructions without any hardcoded logic.
    """
    
    def __init__(self, app_logger):
        self.app_logger = app_logger
        self.llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        self.execution_context = {}
        
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

    async def get_complete_execution_plan(self, pdf_content: str) -> Dict[str, Any]:
        """
        LLM analyzes PDF and creates complete execution plan with all details.
        ZERO hardcoding - LLM extracts everything from PDF including endpoint rules.
        """
        planning_prompt = f"""
You are an intelligent challenge solver. Analyze this PDF document and extract EVERYTHING needed to solve the challenge.

PDF CONTENT:
{pdf_content}

YOUR TASK:
1. Read the PDF completely and understand what challenge it's asking you to solve
2. Extract ALL data structures, mappings, relationships, and rules mentioned in the PDF
3. Extract ALL API endpoints, URL patterns, and service references from the PDF
4. Extract ALL rules about which APIs to call based on what conditions
5. Create a complete execution plan using ONLY information from the PDF

CRITICAL: DO NOT assume anything. Extract everything from the PDF text.

OUTPUT FORMAT (STRICT JSON):
{{
    "challenge_understanding": "What is the challenge asking for based on PDF content?",
    "extracted_data": {{
        "all_mappings": {{"key": "value"}},
        "all_api_endpoints": ["every URL mentioned in PDF"],
        "all_rules_and_conditions": ["every rule mentioned in PDF"],
        "endpoint_selection_rules": {{"condition": "which_endpoint_to_use"}},
        "data_extraction_rules": ["how to extract data from responses"],
        "any_other_patterns": {{"pattern": "value"}}
    }},
    "execution_plan": [
        {{
            "step_id": "step_X",
            "action_type": "API_CALL|DATA_LOOKUP|DATA_PROCESSING|FINAL_RESULT",
            "description": "What this step does based on PDF requirements",
            "details": {{
                "method": "GET|POST",
                "url": "exact URL from PDF or pattern with {{variables}}",
                "params": {{}},
                "response_processing": "how to extract data based on PDF instructions",
                "store_as": "variable_name",
                "lookup_in": "data_source_name",
                "lookup_key": "{{variable_name}}",
                "lookup_store_as": "result_variable",
                "processing_logic": "what to do with data",
                "final_output": "{{final_variable}}"
            }},
            "depends_on": ["previous_steps"],
            "success_criteria": "how to know this worked"
        }}
    ],
    "final_answer_location": "which_step_or_variable"
}}

VARIABLE SYNTAX:
- Use {{variable_name}} to reference stored variables
- Example: {{favorite_city}}, {{landmark}}, {{flight_number}}

RULES:
- Extract ALL information from the PDF text only
- Do NOT assume any standard patterns or common knowledge
- If PDF mentions specific API endpoints, use those exact URLs
- If PDF mentions rules like "if X then call Y", extract those exactly
- Create execution steps based purely on PDF instructions
- Return ONLY valid JSON

Read the PDF carefully and extract everything mentioned in it.
"""
        
        try:
            self.app_logger.info("🧠 LLM analyzing PDF and creating execution plan...")
            result = await self.llm.ainvoke(planning_prompt)
            response_text = result.content.strip()
            self.app_logger.info(f"response:  {response_text}")
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            execution_plan = json.loads(response_text)
            
            self.app_logger.info(f"✅ LLM created plan with {len(execution_plan.get('execution_plan', []))} steps")
            self.app_logger.info(f"🎯 Challenge: {execution_plan.get('challenge_understanding', 'Unknown')}")
            
            return execution_plan
            
        except json.JSONDecodeError as e:
            self.app_logger.error(f"❌ Failed to parse LLM execution plan: {e}")
            self.app_logger.error(f"Raw response: {response_text}")
            raise RuntimeError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            self.app_logger.error(f"❌ LLM planning failed: {e}")
            raise RuntimeError(f"Could not create execution plan: {str(e)}")

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

    async def process_response_with_llm(self, response_data: Any, processing_instruction: str) -> Any:
        """Use LLM to process API response according to instructions"""
        processing_prompt = f"""
Process this API response data according to the given instruction.

RESPONSE DATA: {json.dumps(response_data) if response_data else str(response_data)}

PROCESSING INSTRUCTION: {processing_instruction}

RULES:
- Follow the instruction exactly
- Extract only the requested data
- Return the processed result directly (no explanation)
- If instruction asks for a specific field, return only that field value
- If instruction asks for transformation, return the transformed data

Examples:
- Instruction: "Extract city name" -> Return: "Mumbai"
- Instruction: "Extract flight number" -> Return: "AI123"
- Instruction: "Get the value of 'data.city' field" -> Return: "Delhi"
"""
        
        try:
            result = await self.llm.ainvoke(processing_prompt)
            processed_data = result.content.strip().strip('"\'')
            self.app_logger.info(f"🔄 Processed response: {str(processed_data)[:100]}")
            return processed_data
        except Exception as e:
            self.app_logger.error(f"❌ Response processing failed: {e}")
            return response_data

    async def perform_data_lookup(self, lookup_instruction: str, data_source: Dict, lookup_key: str) -> Any:
        """Use LLM to perform data lookup according to instructions"""
        
        # First try direct lookup without LLM for exact matches
        if isinstance(data_source, dict) and lookup_key in data_source:
            direct_result = data_source[lookup_key]
            self.app_logger.info(f"🔍 Direct lookup success: '{lookup_key}' -> '{direct_result}'")
            return direct_result
        
        # If direct lookup fails, use LLM for fuzzy matching
        lookup_prompt = f"""
Find the value for the given key in the data source.

LOOKUP KEY: {lookup_key}
DATA SOURCE: {json.dumps(data_source, indent=2)}

RULES:
1. Look for EXACT key match first
2. If not found, try case-insensitive matching
3. If still not found, try partial/fuzzy matching
4. For city names, consider variations (e.g., "New York" vs "New York City")
5. Return ONLY the found value (no explanation)
6. If absolutely not found, return "NOT_FOUND"

EXAMPLES:
- Key: "New York", Data: {{"New York": "Eiffel Tower"}} -> Return: "Eiffel Tower"
- Key: "mumbai", Data: {{"Mumbai": "India Gate"}} -> Return: "India Gate"
- Key: "NYC", Data: {{"New York": "Statue of Liberty"}} -> Return: "Statue of Liberty"

Find the value for: {lookup_key}
"""
        
        try:
            result = await self.llm.ainvoke(lookup_prompt)
            lookup_result = result.content.strip().strip('"\'')
            
            if lookup_result == "NOT_FOUND":
                self.app_logger.warning(f"🔍 Lookup failed: No match for '{lookup_key}' in data source")
                # Log available keys for debugging
                if isinstance(data_source, dict):
                    available_keys = list(data_source.keys())[:10]  # First 10 keys
                    self.app_logger.info(f"📋 Available keys: {available_keys}")
                return None
            
            self.app_logger.info(f"🔍 LLM lookup success: '{lookup_key}' -> '{lookup_result}'")
            return lookup_result
        except Exception as e:
            self.app_logger.error(f"❌ Data lookup failed: {e}")
            return None

    async def execute_step(self, step: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step according to LLM instructions"""
        step_id = step.get("step_id", "unknown")
        action_type = step.get("action_type", "")
        description = step.get("description", "")
        details = step.get("details", {})
        
        self.app_logger.info(f"🔧 Executing {step_id}: {description}")
        
        try:
            if action_type == "API_CALL":
                # Execute API call as specified by LLM
                method = details.get("method", "GET")
                url = details.get("url", "")
                params = details.get("params", {})
                
                # If URL is N/A or empty, try to determine it using processing_logic
                if url in ["N/A", "", "None"]:
                    processing_logic = details.get("processing_logic", "")
                    if processing_logic:
                        # Use LLM to determine the URL based on processing logic
                        url_determination_prompt = f"""
Determine the API call details based on the processing logic and available data.

PROCESSING LOGIC: {processing_logic}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}
EXTRACTED DATA: {json.dumps(extracted_data, indent=2)}

RULES:
- Use the processing logic to determine which API endpoint to call
- Look at the execution context for variables like landmark, city, etc.
- Use the extracted data rules to map variables to endpoints
- Return the HTTP method and URL in format: "METHOD URL"

EXAMPLES:
- If landmark is "Eiffel Tower" and rules say "Eiffel Tower -> getThirdCityFlightNumber"
- Return: GET https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber

Return in format: METHOD URL
"""
                        
                        try:
                            url_result = await self.llm.ainvoke(url_determination_prompt)
                            llm_response = url_result.content.strip().strip('"\'')
                            self.app_logger.info(f"🔗 LLM response: {llm_response}")
                            
                            # Parse method and URL from LLM response
                            if llm_response.startswith(("GET ", "POST ", "PUT ", "DELETE ")):
                                # LLM returned "METHOD URL" format
                                parts = llm_response.split(" ", 1)
                                if len(parts) == 2:
                                    determined_method = parts[0]
                                    determined_url = parts[1]
                                    method = determined_method  # Override the method too
                                    url = determined_url
                                    self.app_logger.info(f"🔗 Parsed method: {method}, URL: {url}")
                                else:
                                    url = llm_response
                            else:
                                # LLM returned just URL
                                url = llm_response
                                
                        except Exception as e:
                            self.app_logger.error(f"❌ URL determination failed: {e}")
                            return {
                                "step_id": step_id,
                                "action_type": action_type,
                                "result": None,
                                "success": False,
                                "error": f"Could not determine API URL: {str(e)}"
                            }
                
                # Replace variables in URL and params if needed
                url = await self.replace_variables_in_text(url, extracted_data)
                if params:
                    params = await self.replace_variables_in_dict(params, extracted_data)
                
                response = await self.execute_api_call(method, url, params)
                
                # Process response according to LLM instructions
                processing_instruction = details.get("response_processing", "")
                if processing_instruction:
                    processed_data = await self.process_response_with_llm(response["data"], processing_instruction)
                else:
                    processed_data = response["data"]
                
                # Store result if specified
                store_as = details.get("store_as", "")
                if store_as:
                    self.execution_context[store_as] = processed_data
                    self.app_logger.info(f"💾 Stored '{processed_data}' as '{store_as}'")
                
                return {
                    "step_id": step_id,
                    "action_type": action_type,
                    "result": processed_data,
                    "success": response["success"],
                    "raw_response": response
                }
                
            elif action_type == "DATA_LOOKUP":
                # Perform data lookup as specified by LLM
                lookup_instruction = details.get("lookup_in", "")
                lookup_key_template = details.get("lookup_key", "")
                
                # Replace variables in lookup key
                lookup_key = await self.replace_variables_in_text(lookup_key_template, extracted_data)
                self.app_logger.info(f"🔍 Looking up: '{lookup_key}' (from template: '{lookup_key_template}')")
                
                # Determine data source
                data_source = extracted_data.get("all_mappings", {})
                if lookup_instruction and lookup_instruction in extracted_data:
                    data_source = extracted_data[lookup_instruction]
                
                # Check if we need to reverse the mapping (city->landmark vs landmark->city)
                if lookup_key not in data_source and isinstance(data_source, dict):
                    # Try reversing the mapping
                    reversed_data = {v: k for k, v in data_source.items()}
                    if lookup_key in reversed_data:
                        self.app_logger.info(f"🔄 Using reversed mapping for lookup")
                        data_source = reversed_data
                    else:
                        # Try both original and reversed with LLM
                        self.app_logger.info(f"🔄 Trying both original and reversed mappings")
                        combined_data = {**data_source, **reversed_data}
                        data_source = combined_data
                
                # Log the data source for debugging
                self.app_logger.info(f"📊 Data source keys: {list(data_source.keys()) if isinstance(data_source, dict) else 'Not a dict'}")
                
                lookup_result = await self.perform_data_lookup(
                    f"Find value for key '{lookup_key}'", 
                    data_source, 
                    lookup_key
                )
                
                # Store result if specified
                store_as = details.get("lookup_store_as", "")
                if store_as:
                    self.execution_context[store_as] = lookup_result
                    self.app_logger.info(f"💾 Stored lookup result '{lookup_result}' as '{store_as}'")
                
                return {
                    "step_id": step_id,
                    "action_type": action_type,
                    "result": lookup_result,
                    "success": lookup_result is not None
                }
                
            elif action_type == "DATA_PROCESSING":
                # Process data according to LLM instructions
                processing_logic = details.get("processing_logic", "")
                
                processing_result = await self.process_response_with_llm(
                    self.execution_context, 
                    processing_logic
                )
                
                # Store result if specified
                store_as = details.get("store_as", "")
                if store_as:
                    self.execution_context[store_as] = processing_result
                    self.app_logger.info(f"💾 Stored processed result '{processing_result}' as '{store_as}'")
                
                return {
                    "step_id": step_id,
                    "action_type": action_type,
                    "result": processing_result,
                    "success": True
                }
                
            elif action_type == "FINAL_RESULT":
                # Return final result as specified by LLM
                final_output_template = details.get("final_output", "")
                
                if final_output_template:
                    # Replace variables in final output template
                    final_result = await self.replace_variables_in_text(final_output_template, extracted_data)
                    self.app_logger.info(f"🎯 Final result from template '{final_output_template}' -> '{final_result}'")
                else:
                    # Use the last stored result
                    final_result = list(self.execution_context.values())[-1] if self.execution_context else "No result"
                    self.app_logger.info(f"🎯 Final result from last context value: '{final_result}'")
                
                return {
                    "step_id": step_id,
                    "action_type": action_type,
                    "result": final_result,
                    "success": True,
                    "is_final": True
                }
                
        except Exception as e:
            self.app_logger.error(f"❌ Step {step_id} execution failed: {e}")
            return {
                "step_id": step_id,
                "action_type": action_type,
                "result": None,
                "success": False,
                "error": str(e)
            }

    async def replace_variables_in_dict(self, data: Dict, extracted_data: Dict[str, Any] = None) -> Dict:
        """Replace variables in dictionary values"""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = await self.replace_variables_in_text(value, extracted_data)
            else:
                result[key] = value
        return result
            

    async def replace_variables_in_text(self, text: str, extracted_data: Dict[str, Any] = None) -> str:
        """Replace variables in text with values from execution context"""
        if not isinstance(text, str):
            return text
            
        # Replace variables like {variable_name} with actual values
        for var_name, var_value in self.execution_context.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(var_value))
        
        # Handle any remaining variables using pure LLM resolution
        if "{" in text and "}" in text:
            text = await self.resolve_remaining_variables(text, extracted_data)
        
        return text

    async def resolve_remaining_variables(self, text: str, extracted_data: Dict[str, Any]) -> str:
        """Use LLM to resolve any remaining variables using extracted data"""
        resolve_prompt = f"""
You are a variable resolution specialist. Resolve the variables in this text using the available data.

TEXT WITH VARIABLES: {text}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}
EXTRACTED DATA: {json.dumps(extracted_data, indent=2)}

YOUR TASK:
1. Identify what variable is being requested in the text
2. Look through the execution context to find the most appropriate value
3. Consider the challenge context to determine what the final answer should be
4. Return ONLY the resolved value, not explanations or metadata

RULES:
- If asking for a final result, find the most relevant result from execution context
- Match variable names intelligently (final_flight_number could mean flight_number)
- Consider the challenge type to determine what kind of answer is expected
- Return the actual data value, not URLs, descriptions, or metadata
- If multiple possible values exist, choose the one that best fits the challenge objective
- If resolving an API endpoint, return ONLY the URL (no HTTP method)

EXAMPLES:
- Text: "{{final_flight_number}}", Context: {{"flight_number": "09811e"}} → Return: "09811e"
- Text: "{{product_code}}", Context: {{"code": "ABC123"}} → Return: "ABC123"  
- Text: "{{endpoint_selection_rules[landmark]}}", Rules: {{"Eiffel Tower": "GET https://api.com/endpoint"}} → Return: "https://api.com/endpoint"

Analyze the context and return the most appropriate value for: {text}
"""
        
        try:
            result = await self.llm.ainvoke(resolve_prompt)
            resolved_text = result.content.strip().strip('"\'')
            
            # If the resolved text contains "METHOD URL", extract just the URL
            if resolved_text.startswith(("GET ", "POST ", "PUT ", "DELETE ")):
                parts = resolved_text.split(" ", 1)
                if len(parts) == 2:
                    resolved_text = parts[1]  # Just the URL part
                    self.app_logger.info(f"🔧 Extracted URL from method+URL: {resolved_text}")
            
            self.app_logger.info(f"🔧 LLM resolved '{text}' -> '{resolved_text}'")
            return resolved_text
        except Exception as e:
            self.app_logger.error(f"❌ Variable resolution failed: {e}")
            return text

    async def solve_challenge(self, doc_url: str) -> str:
        """
        Main method to solve challenge using pure LLM-driven approach.
        LLM analyzes PDF and provides complete execution plan.
        """
        try:
            # 1. Download PDF
            parsed = urlparse(doc_url)
            path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
            ext = path_ext if path_ext else "pdf"
            file_path, _doc_hash = await download_and_hash_document(doc_url, ext, self.app_logger)
            
            # 2. Extract PDF content
            pdf_content = await self.extract_pdf_content(file_path)
            
            # 3. Get complete execution plan from LLM
            execution_plan = await self.get_complete_execution_plan(pdf_content)
            
            extracted_data = execution_plan.get("extracted_data", {})
            steps = execution_plan.get("execution_plan", [])
            final_answer_location = execution_plan.get("final_answer_location", "")
            
            # 4. Execute steps according to LLM plan
            final_result = None
            step_results = []
            
            for step in steps:
                # Check dependencies
                depends_on = step.get("depends_on", [])
                if depends_on:
                    # Verify all dependencies are satisfied
                    for dep in depends_on:
                        if not any(r.get("step_id") == dep and r.get("success") for r in step_results):
                            self.app_logger.warning(f"⚠️ Dependency {dep} not satisfied for {step.get('step_id')}")
                
                step_result = await self.execute_step(step, extracted_data)
                step_results.append(step_result)
                
                # Check if this is the final result
                if step_result.get("is_final") or step.get("step_id") == final_answer_location:
                    final_result = step_result["result"]
                    break
                
                # Small delay between steps
                await asyncio.sleep(0.5)
            
            # 5. Determine final answer
            if not final_result:
                if final_answer_location and final_answer_location in self.execution_context:
                    final_result = self.execution_context[final_answer_location]
                else:
                    # Use LLM to determine final answer from execution context
                    final_prompt = f"""
Based on the execution results, determine the final answer to the challenge.

CHALLENGE: {execution_plan.get('challenge_understanding', 'Unknown')}
EXECUTION CONTEXT: {json.dumps(self.execution_context, indent=2)}
STEP RESULTS: {json.dumps([r for r in step_results if r.get("success")], indent=2)}

What is the final answer? Return ONLY the answer value.
"""
                    
                    final_llm_result = await self.llm.ainvoke(final_prompt)
                    final_result = final_llm_result.content.strip()
            
            self.app_logger.info(f"✅ Challenge solved: {final_result}")
            return str(final_result)
            
        except Exception as e:
            self.app_logger.error(f"❌ LLM-driven challenge solving failed: {e}")
            raise RuntimeError(f"Challenge agent failed: {str(e)}")


# Main handler function
async def handle_hackrx_challenge(doc_url: str, app_logger, executor) -> str:
    """
    Handle HackRX challenge using pure LLM-driven approach.
    LLM analyzes PDF and provides complete execution instructions.
    """
    try:
        app_logger.info("🧠 Starting LLM-driven challenge solving...")
        agent = LLMDrivenChallengeAgent(app_logger)
        result = await agent.solve_challenge(doc_url)
        app_logger.info(f"✅ LLM-driven agent solved challenge: {result}")
        return result
        
    except Exception as e:
        app_logger.error(f"❌ LLM-driven challenge solving failed: {e}")
        raise RuntimeError(f"Challenge agent failed: {str(e)}")
