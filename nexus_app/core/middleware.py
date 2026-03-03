import json
import time
from typing import Any, Dict

from fastapi import Request


def create_request_logger_middleware(app_logger):
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        headers = dict(request.headers)

        request_details: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "url": url,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else {},
            "headers": headers,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "Unknown"),
        }

        app_logger.info(f"🌐 INCOMING REQUEST: {method} {url}")
        app_logger.info(f"🔍 Client IP: {client_ip}")
        app_logger.info(f"📋 Headers: {json.dumps(headers, indent=2)}")

        if method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    try:
                        body_str = body.decode("utf-8")
                        if request.headers.get("content-type", "").startswith("application/json"):
                            try:
                                body_json = json.loads(body_str)
                                request_details["body"] = body_json
                                app_logger.info("📝 Complete Request Body (JSON):")
                                app_logger.info(json.dumps(body_json, indent=2, ensure_ascii=False))
                                if isinstance(body_json, dict):
                                    if "questions" in body_json:
                                        app_logger.info(f"❓ Questions Count: {len(body_json.get('questions', []))}")
                                        for i, q in enumerate(body_json.get("questions", []), 1):
                                            app_logger.info(f"   Q{i}: {q}")
                                    if "documents" in body_json:
                                        app_logger.info(f"📄 Document URL: {body_json.get('documents', 'N/A')}")
                            except json.JSONDecodeError:
                                request_details["body"] = body_str
                                app_logger.info(f"📝 Complete Request Body (Invalid JSON):\n{body_str}")
                        else:
                            request_details["body"] = body_str
                            app_logger.info(f"📝 Complete Request Body (Raw):\n{body_str}")
                    except UnicodeDecodeError:
                        request_details["body"] = f"<Binary data: {len(body)} bytes>"
                        app_logger.info(f"📝 Complete Request Body (Binary): {len(body)} bytes")
                        hex_preview = body[:500].hex()
                        app_logger.info(f"📝 Binary Preview (hex): {hex_preview}")
                else:
                    request_details["body"] = None
                    app_logger.info("📝 Request Body: Empty")
            except Exception as e:
                request_details["body_error"] = str(e)
                app_logger.warning(f"⚠️ Could not read request body: {e}")

        if request.query_params:
            app_logger.info(f"🔗 Query Parameters: {dict(request.query_params)}")

        app_logger.info("🔍 COMPLETE REQUEST STRUCTURE:")
        app_logger.info(json.dumps(request_details, indent=2, ensure_ascii=False))

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response_details = {
                "status_code": response.status_code,
                "processing_time": f"{process_time:.2f}s",
                "response_headers": dict(response.headers),
            }
            app_logger.info(f"✅ RESPONSE: {response.status_code} | Time: {process_time:.2f}s")
            app_logger.info(f"📤 Response Details: {json.dumps(response_details, indent=2)}")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            error_details = {
                "error": str(e),
                "processing_time": f"{process_time:.2f}s",
                "error_type": type(e).__name__,
            }
            app_logger.error(f"❌ REQUEST FAILED: {str(e)} | Time: {process_time:.2f}s")
            app_logger.error(f"❌ ERROR DETAILS: {json.dumps(error_details, indent=2)}")
            app_logger.exception("Full error traceback:")
            raise

    return log_requests


