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

from nexus_app.services.document_io import download_and_hash_document
from langchain_openai import ChatOpenAI
from nexus_app.utils.constants import GPT_PRIMARY



class NexusChallengeAgent:
    """
    Intelligent agent that solves Nexus challenges by analyzing PDFs and executing API calls.
    No hardcoded data - everything is extracted dynamically from the PDF.
    """

    # Class-level in-memory cache for extracted data and strategy, keyed by doc hash
    _llm_cache = {}
    _cache_dir = '.llm_cache'

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


    async def solve_challenge(self, doc_url: str) -> str:
        """
        Main method to solve the Nexus challenge using pure agent-based approach.
        No hardcoded data - everything extracted from PDF.
        Uses persistent caching for LLM extraction/plan if the same doc is given as input.
        """
        import os
        try:
            # 1. Download PDF
            parsed = urlparse(doc_url)
            path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
            ext = path_ext if path_ext else "pdf"
            file_path, doc_hash = await download_and_hash_document(doc_url, ext, self.app_logger)

            # 2. Persistent cache dir
            cache_key = doc_hash or doc_url
            cache_dir = self._cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")

            cache = None
            # Try in-memory cache first
            if cache_key in self._llm_cache:
                cache = self._llm_cache[cache_key]
            # Try disk cache if not in memory
            elif os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache = json.load(f)
                        self._llm_cache[cache_key] = cache
                    self.app_logger.info(f"♻️ Loaded LLM extraction/plan from disk cache: {cache_path}")
                except Exception as e:
                    self.app_logger.error(f"❌ Failed to load cache from {cache_path}: {e}")

            if cache:
                self.app_logger.info("♻️ Using cached LLM extraction and strategy for this document.")
                self.extracted_data = cache["extracted_data"]
                strategy = cache["strategy"]
            else:
                # 2. Extract PDF content
                pdf_content = await self.extract_pdf_content(file_path)

                # 3. Extract structured data from PDF (replaces hardcoded mappings)
                self.extracted_data = await self.extract_structured_data_from_pdf(pdf_content)

                # 4. Analyze challenge strategy
                strategy = await self.analyze_challenge_strategy(pdf_content)

                # Store in both memory and disk cache
                cache_obj = {
                    "extracted_data": self.extracted_data,
                    "strategy": strategy
                }
                self._llm_cache[cache_key] = cache_obj
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_obj, f, ensure_ascii=False, indent=2)
                    self.app_logger.info(f"💾 Saved LLM extraction/plan to disk cache: {cache_path}")
                except Exception as e:
                    self.app_logger.error(f"❌ Failed to save cache to {cache_path}: {e}")

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
async def handle_nexus_challenge(doc_url: str, app_logger, executor) -> str:
    """
    Handle Nexus challenge using the intelligent agent approach.
    No hardcoded data - everything is extracted dynamically from PDF.
    """
    agent = NexusChallengeAgent(app_logger)
    return await agent.solve_challenge(doc_url)
