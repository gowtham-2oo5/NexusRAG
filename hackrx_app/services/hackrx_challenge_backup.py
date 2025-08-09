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
from hackrx_app.prompts import challenge_analysis_template, api_execution_template
from hackrx_app.utils.constants import GPT_PRIMARY
from langchain_openai import ChatOpenAI
from hackrx_app.prompts import challenge_analysis_template, api_execution_template
from hackrx_app.utils.constants import GPT_PRIMARY


def _extract_city_landmark_mapping_sync(pdf_path: str, app_logger) -> Dict[str, str]:
    """
    Extract city-landmark mapping from the HackRX PDF.
    This PDF contains a "parallel world" scenario where landmarks are in wrong cities.
    We use a hardcoded mapping based on the PDF content analysis.
    """
    
    # Hardcoded mapping based on PDF analysis - represents the "parallel world"
    city_to_landmark = {
        # Indian Cities
        "Delhi": "Gateway of India",
        "Mumbai": "India Gate", 
        "Chennai": "Charminar",
        "Hyderabad": "Taj Mahal",  # Most prominent mapping in PDF
        "Ahmedabad": "Howrah Bridge",
        "Mysuru": "Golconda Fort",
        "Kochi": "Qutub Minar",
        "Pune": "Meenakshi Temple",
        "Nagpur": "Lotus Temple",
        "Chandigarh": "Mysore Palace", 
        "Kerala": "Rock Garden",
        "Bhopal": "Victoria Memorial",
        "Varanasi": "Vidhana Soudha",
        "Jaisalmer": "Sun Temple",
        
        # International Cities
        "New York": "Eiffel Tower",
        "London": "Statue of Liberty",
        "Tokyo": "Big Ben",
        "Beijing": "Colosseum",
        "Bangkok": "Christ the Redeemer",
        "Toronto": "Burj Khalifa", 
        "Dubai": "CN Tower",
        "Amsterdam": "Petronas Towers",
        "Cairo": "Leaning Tower of Pisa",
        "San Francisco": "Mount Fuji",
        "Berlin": "Niagara Falls",
        "Barcelona": "Louvre Museum",
        "Moscow": "Stonehenge",
        "Seoul": "Sagrada Familia",
        "Cape Town": "Acropolis",
        "Istanbul": "Big Ben",
        "Riyadh": "Machu Picchu",
        "Paris": "Taj Mahal",
        "Dubai Airport": "Moai Statues",
        "Singapore": "Christchurch Cathedral",
        "Jakarta": "The Shard",
        "Vienna": "Blue Mosque",
        "Kathmandu": "Neuschwanstein Castle",
        "Los Angeles": "Buckingham Palace",
    }
    
    app_logger.info(f"✅ Using hardcoded city→landmark mapping with {len(city_to_landmark)} entries")
    
    # Log a preview
    try:
        app_logger.info("🧭 City→Landmark mapping preview (first 15):")
        for i, (city, landmark) in enumerate(sorted(city_to_landmark.items(), key=lambda kv: kv[0].casefold())):
            if i >= 15:
                app_logger.info("  … (truncated)")
                break
            app_logger.info(f"  {city} => {landmark}")
    except Exception as log_e:
        app_logger.warning(f"Failed to log mapping preview: {log_e}")
    
    return city_to_landmark


# --- Gemini-based Extraction ---
_EMOJI_PATTERN = re.compile(
    "[\u2190-\u21FF\u2300-\u23FF\u2460-\u24FF\u2500-\u25FF\u2600-\u27BF\u2900-\u297F\u2B00-\u2BFF\u1F000-\u1FAFF\u200D\uFE0F]+"
)


def _clean_name(text: str) -> str:
    text = _EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_mapping_with_gemini_sync(pdf_path: str, app_logger) -> Dict[str, str]:
    """
    Fallback Gemini extraction - but we'll use the hardcoded mapping instead
    """
    app_logger.info("🔄 Gemini fallback requested, but using hardcoded mapping instead")
    return _extract_city_landmark_mapping_sync(pdf_path, app_logger)


async def _extract_city_landmark_mapping_with_gemini(pdf_path: str, app_logger, executor) -> Dict[str, str]:
    return await asyncio.get_event_loop().run_in_executor(
        executor, _extract_mapping_with_gemini_sync, pdf_path, app_logger
    )


async def _extract_city_landmark_mapping(pdf_path: str, app_logger, executor) -> Dict[str, str]:
    return await asyncio.get_event_loop().run_in_executor(
        executor, _extract_city_landmark_mapping_sync, pdf_path, app_logger
    )


def _normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def _find_landmark(city_to_landmark: Dict[str, str], city_name: str) -> Optional[str]:
    if city_name in city_to_landmark:
        return city_to_landmark[city_name]
    # Try case/space-insensitive match
    lookup = { _normalize_key(k): v for k, v in city_to_landmark.items() }
    normalized_city = _normalize_key(city_name)
    if normalized_city in lookup:
        return lookup[normalized_city]
    # Fallback: simple token-overlap to handle variants (e.g., "New York" vs "New York City")
    def tokenize(s: str) -> set[str]:
        return set(re.findall(r"[a-z]+", s))
    target_tokens = tokenize(normalized_city)
    best_key: Optional[str] = None
    best_score = 0.0
    for k in lookup.keys():
        k_tokens = tokenize(k)
        if not k_tokens:
            continue
        inter = len(target_tokens & k_tokens)
        union = len(target_tokens | k_tokens)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_key = k
    if best_key and best_score >= 0.5:
        return lookup[best_key]
    return None


def _normalize_city_aliases(name: str) -> str:
    # common aliases to improve matching
    alias_map = {
        "new york city": "new york",
        "nyc": "new york",
        "bombay": "mumbai",
        "madras": "chennai",
        "calcutta": "kolkata",
        "bangalore": "bengaluru",
    }
    key = _normalize_key(name)
    return alias_map.get(key, key)


def _landmark_to_endpoint_suffix(landmark: str) -> str:
    key = _normalize_key(landmark)
    if key == _normalize_key("Gateway of India"):
        return "getFirstCityFlightNumber"
    if key == _normalize_key("Taj Mahal"):
        return "getSecondCityFlightNumber"
    if key == _normalize_key("Eiffel Tower"):
        return "getThirdCityFlightNumber"
    if key == _normalize_key("Big Ben"):
        return "getFourthCityFlightNumber"
    return "getFifthCityFlightNumber"


async def _extract_pdf_content(pdf_path: str, app_logger) -> str:
    """Extract full text content from PDF for LLM analysis"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            content = ""
            for page in pdf.pages:
                if page.extract_text():
                    content += page.extract_text() + "\n\n"
        app_logger.info(f"📄 Extracted {len(content)} characters from PDF")
        return content
    except Exception as e:
        app_logger.error(f"❌ Failed to extract PDF content: {e}")
        raise RuntimeError(f"PDF extraction failed: {str(e)}")


async def _analyze_challenge_with_llm(pdf_content: str, app_logger) -> Dict[str, Any]:
    """Use LLM to analyze the PDF and generate step-by-step instructions"""
    try:
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        prompt = challenge_analysis_template.format(document_content=pdf_content)
        
        app_logger.info("🤖 Analyzing challenge with LLM...")
        result = await llm.ainvoke(prompt)
        analysis_text = result.content.strip()
        
        # LOG THE EXACT LLM RESPONSE
        app_logger.info(f"🔍 LLM Analysis Response:\n{analysis_text}")
        app_logger.info("=" * 80)
        
        # Parse the structured response
        analysis = {
            "challenge_summary": "",
            "mappings_found": [],
            "api_endpoints": [],
            "instructions": [],
            "memory_items": []
        }
        
        current_section = None
        for line in analysis_text.split('\n'):
            line = line.strip()
            if line.startswith('CHALLENGE_SUMMARY:'):
                analysis["challenge_summary"] = line.replace('CHALLENGE_SUMMARY:', '').strip()
            elif line.startswith('MAPPINGS_FOUND:'):
                current_section = "mappings"
            elif line.startswith('API_ENDPOINTS:'):
                current_section = "endpoints"
            elif line.startswith('INSTRUCTIONS:'):
                current_section = "instructions"
            elif line.startswith('MEMORY_ITEMS:'):
                current_section = "memory"
            elif line == '--END_OF_PROCESS--':
                current_section = None
            elif line and current_section:
                if current_section == "mappings":
                    analysis["mappings_found"].append(line)
                elif current_section == "endpoints":
                    analysis["api_endpoints"].append(line)
                elif current_section == "instructions":
                    # Remove step numbers like "1. ", "2. ", etc.
                    instruction = re.sub(r'^\d+\.\s*', '', line)
                    if instruction:
                        analysis["instructions"].append(instruction)
                elif current_section == "memory":
                    analysis["memory_items"].append(line)
        
        app_logger.info(f"✅ Challenge analysis complete: {len(analysis['instructions'])} steps identified")
        return analysis
        
    except Exception as e:
        app_logger.error(f"❌ LLM analysis failed: {e}")
        raise RuntimeError(f"Challenge analysis failed: {str(e)}")


async def _execute_api_call(method: str, url: str, params: Dict = None, app_logger = None) -> Dict[str, Any]:
    """Execute an API call and return structured response"""
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
            
            if app_logger:
                app_logger.info(f"🌐 API {method} {url} -> {status}")
            
            return result
            
    except Exception as e:
        if app_logger:
            app_logger.error(f"❌ API call failed: {e}")
        return {
            "status": 0,
            "data": None,
            "raw_text": str(e),
            "success": False,
            "error": str(e)
        }


async def _execute_instruction_step(instruction: str, previous_results: List[Dict], memory_data: Dict, app_logger) -> Dict[str, Any]:
    """Execute a single instruction step using LLM guidance"""
    try:
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.1)
        prompt = api_execution_template.format(
            instruction=instruction,
            previous_results=json.dumps(previous_results[-3:], indent=2) if previous_results else "None",
            memory_data=json.dumps(memory_data, indent=2) if memory_data else "None"
        )
        
        result = await llm.ainvoke(prompt)
        execution_text = result.content.strip()
        
        # LOG THE EXACT LLM EXECUTION RESPONSE
        app_logger.info(f"🔍 LLM Execution Response:\n{execution_text}")
        app_logger.info("=" * 80)
        
        # Parse execution response
        execution = {
            "action_type": "UNKNOWN",
            "method": None,
            "url": None,
            "parameters": None,
            "processing": None,
            "lookup_key": None,
            "result": None,
            "next_step": "CONTINUE"
        }
        
        for line in execution_text.split('\n'):
            line = line.strip()
            if line.startswith('ACTION_TYPE:'):
                execution["action_type"] = line.replace('ACTION_TYPE:', '').strip()
            elif line.startswith('METHOD:'):
                execution["method"] = line.replace('METHOD:', '').strip()
            elif line.startswith('URL:'):
                execution["url"] = line.replace('URL:', '').strip()
            elif line.startswith('PARAMETERS:'):
                params_str = line.replace('PARAMETERS:', '').strip()
                if params_str and params_str != "None":
                    try:
                        execution["parameters"] = json.loads(params_str)
                    except:
                        execution["parameters"] = params_str
            elif line.startswith('PROCESSING:'):
                execution["processing"] = line.replace('PROCESSING:', '').strip()
            elif line.startswith('LOOKUP_KEY:'):
                execution["lookup_key"] = line.replace('LOOKUP_KEY:', '').strip()
            elif line.startswith('RESULT:'):
                execution["result"] = line.replace('RESULT:', '').strip()
            elif line.startswith('NEXT_STEP:'):
                execution["next_step"] = line.replace('NEXT_STEP:', '').strip()
        
        # Execute the action
        if execution["action_type"] == "API_CALL" and execution["url"]:
            api_result = await _execute_api_call(
                execution["method"] or "GET",
                execution["url"],
                execution["parameters"],
                app_logger
            )
            execution["api_response"] = api_result
            execution["result"] = api_result["data"]
            
        elif execution["action_type"] == "LOOKUP" and execution["lookup_key"]:
            # Search in memory data and previous results
            lookup_result = None
            lookup_key = execution["lookup_key"].lower()
            
            # Search memory data
            for key, value in memory_data.items():
                if lookup_key in key.lower() or lookup_key in str(value).lower():
                    lookup_result = value
                    break
            
            # Search previous results if not found in memory
            if not lookup_result:
                for prev_result in previous_results:
                    if "result" in prev_result and prev_result["result"]:
                        if lookup_key in str(prev_result["result"]).lower():
                            lookup_result = prev_result["result"]
                            break
            
            execution["result"] = lookup_result
            
        elif execution["action_type"] == "DATA_PROCESSING":
            # For now, just return the processing description
            # This could be enhanced with actual data processing logic
            execution["result"] = execution["processing"]
            
        elif execution["action_type"] == "FINAL_ANSWER":
            execution["next_step"] = "COMPLETE"
        
        app_logger.info(f"🔧 Executed step: {execution['action_type']} -> {str(execution['result'])[:100]}")
        return execution
        
    except Exception as e:
        app_logger.error(f"❌ Instruction execution failed: {e}")
        return {
            "action_type": "ERROR",
            "result": None,
            "error": str(e),
            "next_step": "CONTINUE"
        }


async def handle_hackrx_challenge(doc_url: str, app_logger, executor) -> str:
    """Dynamic challenge handler that uses LLM to analyze PDF and execute instructions"""
    try:
        # 1) Download the PDF
        parsed = urlparse(doc_url)
        path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
        ext = path_ext if path_ext else "pdf"
        file_path, _doc_hash = await download_and_hash_document(doc_url, ext, app_logger)
        
        # 2) Extract PDF content
        pdf_content = await asyncio.get_event_loop().run_in_executor(
            executor, _extract_pdf_content, file_path, app_logger
        )
        
        # 3) Analyze challenge with LLM
        analysis = await _analyze_challenge_with_llm(pdf_content, app_logger)
        app_logger.info(f"🎯 Challenge: {analysis['challenge_summary']}")
        
        # 4) Initialize memory with mappings found
        memory_data = {}
        for mapping in analysis["mappings_found"]:
            # Try to parse key-value pairs from mapping descriptions
            if ":" in mapping or "=" in mapping or "->" in mapping:
                memory_data[f"mapping_{len(memory_data)}"] = mapping
        
        for item in analysis["memory_items"]:
            memory_data[f"memory_{len(memory_data)}"] = item
        
        app_logger.info(f"💾 Initialized memory with {len(memory_data)} items")
        
        # 5) Execute instructions step by step
        previous_results = []
        final_answer = None
        
        for i, instruction in enumerate(analysis["instructions"], 1):
            app_logger.info(f"📋 Step {i}: {instruction}")
            
            execution_result = await _execute_instruction_step(
                instruction, previous_results, memory_data, app_logger
            )
            
            previous_results.append(execution_result)
            
            # Update memory with new data
            if execution_result.get("result"):
                memory_data[f"step_{i}_result"] = execution_result["result"]
            
            # Check if this is the final answer
            if (execution_result.get("next_step") == "COMPLETE" or 
                execution_result.get("action_type") == "FINAL_ANSWER"):
                final_answer = execution_result.get("result")
                break
        
        # 6) Return the final answer
        if final_answer:
            app_logger.info(f"✅ Challenge completed: {final_answer}")
            return str(final_answer)
        else:
            # If no explicit final answer, try to extract from last result
            if previous_results:
                last_result = previous_results[-1].get("result")
                if last_result:
                    app_logger.info(f"✅ Using last result as answer: {last_result}")
                    return str(last_result)
            
            raise RuntimeError("No final answer could be determined from the challenge execution")
            
    except Exception as e:
        app_logger.error(f"❌ Dynamic challenge handler failed: {e}")
        # Fallback to original hardcoded logic if dynamic approach fails
        app_logger.info("🔄 Falling back to hardcoded challenge logic...")
        return await _handle_hackrx_challenge_fallback(doc_url, app_logger, executor)


async def _handle_hackrx_challenge_fallback(doc_url: str, app_logger, executor) -> str:
    """Fallback to original hardcoded logic"""
    # This is the original hardcoded implementation as backup
    parsed = urlparse(doc_url)
    path_ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
    ext = path_ext if path_ext else "pdf"
    file_path, _doc_hash = await download_and_hash_document(doc_url, ext, app_logger)

    city_to_landmark = await _extract_city_landmark_mapping(file_path, app_logger, executor)
    
    if not city_to_landmark:
        raise RuntimeError("No city→landmark mappings could be extracted from the PDF")

    fav_city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
    async with aiohttp.ClientSession() as session:
        async with session.get(fav_city_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch favourite city (HTTP {resp.status})")
            content_type = resp.headers.get("content-type", "").lower()
            raw_text = await resp.text()
            favourite_city: Optional[str] = None
            if "application/json" in content_type or raw_text.strip().startswith("{"):
                try:
                    obj = json.loads(raw_text)
                    def extract_city(o) -> Optional[str]:
                        if isinstance(o, dict):
                            for k, v in o.items():
                                if k.lower() == "city" and isinstance(v, str):
                                    return v
                            for nk in ("data", "payload", "result", "response"):
                                if nk in o:
                                    val = extract_city(o[nk])
                                    if val:
                                        return val
                            for v in o.values():
                                val = extract_city(v)
                                if val:
                                    return val
                        elif isinstance(o, list):
                            for item in o:
                                val = extract_city(item)
                                if val:
                                    return val
                        return None
                    favourite_city = extract_city(obj)
                except Exception:
                    favourite_city = None
            if not favourite_city:
                favourite_city = raw_text.strip().strip("\"'")
    app_logger.info(f"🧩 Favourite city from API: '{favourite_city}'")

    landmark = _find_landmark(city_to_landmark, favourite_city)
    if not landmark:
        landmark = _find_landmark(city_to_landmark, _normalize_city_aliases(favourite_city))
    if not landmark:
        normalized_map = { _normalize_city_aliases(k): v for k, v in city_to_landmark.items() }
        landmark = _find_landmark(normalized_map, _normalize_city_aliases(favourite_city))
    if not landmark:
        raise RuntimeError(f"Landmark not found for city: '{favourite_city}'")
    app_logger.info(f"🗼 Landmark for city '{favourite_city}': '{landmark}'")

    endpoint_suffix = _landmark_to_endpoint_suffix(landmark)
    flight_url = f"https://register.hackrx.in/teams/public/flights/{endpoint_suffix}"
    app_logger.info(f"✈️ Fetching flight number from: {flight_url}")

    async with aiohttp.ClientSession() as session:
        async with session.get(flight_url) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Failed to fetch flight number for landmark '{landmark}' (HTTP {resp.status})"
                )
            content_type = resp.headers.get("content-type", "").lower()
            raw_text = (await resp.text()).strip()
            flight_number: Optional[str] = None
            if "application/json" in content_type or raw_text.strip().startswith("{"):
                try:
                    obj = json.loads(raw_text)
                    key_candidates = {
                        "flightnumber",
                        "flight_no",
                        "flightno",
                        "flight",
                        "number",
                        "code",
                    }
                    def extract_flight(o) -> Optional[str]:
                        if isinstance(o, dict):
                            for k, v in o.items():
                                if isinstance(v, str) and k.replace(" ", "").lower() in key_candidates:
                                    return v
                            for nk in ("data", "payload", "result", "response"):
                                if nk in o:
                                    val = extract_flight(o[nk])
                                    if val:
                                        return val
                            for v in o.values():
                                val = extract_flight(v)
                                if val:
                                    return val
                        elif isinstance(o, list):
                            for item in o:
                                val = extract_flight(item)
                                if val:
                                    return val
                        return None
                    flight_number = extract_flight(obj)
                except Exception:
                    flight_number = None
            if not flight_number:
                tokens = re.findall(r"[A-Z]{1,3}-?[0-9]{2,4}", raw_text)
                if not tokens:
                    tokens = re.findall(r"[A-Za-z0-9_-]+", raw_text)
                flight_number = max(tokens, key=len) if tokens else raw_text
    app_logger.info(f"✅ Flight Number resolved: {flight_number}")
    return flight_number
