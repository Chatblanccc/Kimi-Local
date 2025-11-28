from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional
import json
import uuid
import base64
from pathlib import Path

load_dotenv()

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost:5173",  # Frontend dev server
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://192.168.230.2:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Moonshot (Kimi) API 配置
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"

# GPT API 配置 (api.gptsapi.net)
GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_BASE_URL = "https://api.gptsapi.net/v1"

# Gemini API 配置 (通过 api.gptsapi.net 代理)
GEMINI_PROXY_BASE_URL = "https://api.gptsapi.net/api/v3"

# Google 官方 Gemini API 配置
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

if not MOONSHOT_API_KEY:
    print("Warning: MOONSHOT_API_KEY not found in environment variables.")

if not GPT_API_KEY:
    print("Warning: GPT_API_KEY not found in environment variables.")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. Gemini image generation will use proxy API.")
else:
    print("✓ GOOGLE_API_KEY found. Using official Google Gemini API for image generation.")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/google-models")
async def list_google_models():
    """列出 Google API Key 支持的所有模型"""
    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY not configured"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{GOOGLE_GEMINI_BASE_URL}/models?key={GOOGLE_API_KEY}",
            timeout=30.0
        )
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            # 只返回模型名称
            model_names = [m.get("name", "").replace("models/", "") for m in models]
            return {"models": model_names}
        else:
            return {"error": response.text}

@app.get("/api/models")
async def get_available_models():
    """获取可用的模型列表"""
    models = []
    
    # Moonshot (Kimi) 模型
    if MOONSHOT_API_KEY:
        models.extend([
            {"id": "kimi-k2-0905-preview", "name": "Kimi K2 Preview", "provider": "moonshot"},
            {"id": "moonshot-v1-8k", "name": "Moonshot V1 8K", "provider": "moonshot"},
            {"id": "moonshot-v1-32k", "name": "Moonshot V1 32K", "provider": "moonshot"},
            {"id": "moonshot-v1-128k", "name": "Moonshot V1 128K", "provider": "moonshot"},
        ])
    
    # GPT 模型 (api.gptsapi.net)
    if GPT_API_KEY:
        models.extend([
            {"id": "gpt-5.1", "name": "GPT-5.1", "provider": "openai"},
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
        ])
    
    # Google Gemini 模型 (通过代理 API)
    if GPT_API_KEY:
        models.extend([
            {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro", "provider": "google"},
            {"id": "gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro", "provider": "google"},
            {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash", "provider": "google"},
            {"id": "gemini-3-pro-image-preview", "name": "Gemini 3 Pro Image", "provider": "gemini"},
        ])
    
    return {"models": models}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not MOONSHOT_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    try:
        # Read file content
        content = await file.read()
        
        # 使用安全的 ASCII 文件名，避免中文编码问题
        original_filename = file.filename or "file"
        file_ext = Path(original_filename).suffix.lower()
        safe_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # 确保 content_type 是纯 ASCII
        content_type = file.content_type or "application/octet-stream"
        is_image = content_type.startswith("image/") or file_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        
        safe_content_type = "application/octet-stream"
        if is_image:
            ext_mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            safe_content_type = ext_mime_map.get(file_ext, "image/png")
        elif file_ext == ".pdf":
            safe_content_type = "application/pdf"
        
        async with httpx.AsyncClient() as client:
            # 手动构建 multipart 数据，避免编码问题
            files_data = {
                "file": (safe_filename, content, safe_content_type),
                "purpose": (None, "file-extract"),
            }
            
            response = await client.post(
                f"{MOONSHOT_BASE_URL}/files",
                headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"},
                files=files_data,
                timeout=60.0
            )
            
            if response.status_code != 200:
                print(f"Moonshot Upload Error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to upload to Moonshot")
            
            result = response.json()
            return {"file_id": result["id"], "filename": file.filename}
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_and_parse")
async def upload_and_parse(
    file: UploadFile = File(...),
    model: str = Form(default="kimi-k2-0905-preview")
):
    """
    上传文件并解析内容
    - 对于 Kimi 模型：使用 Moonshot API 上传并提取文本
    - 对于 GPT/DALL-E/Gemini 模型：返回 base64 编码的图片数据（用于多模态/图像编辑）
    """
    try:
        content = await file.read()
        original_filename = file.filename or "file"
        file_ext = Path(original_filename).suffix.lower()
        content_type = file.content_type or "application/octet-stream"
        
        # 判断是否是图片文件
        is_image = content_type.startswith("image/") or file_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        
        # 调试日志
        print(f"[DEBUG] model={model}, is_image={is_image}, file_ext={file_ext}, content_type={content_type}")
        
        # 对于 GPT/DALL-E/Gemini 模型且是图片，返回 base64 数据
        if (model.startswith("gpt") or model.startswith("dall-e") or model.startswith("gemini")) and is_image:
            print(f"Processing image for model {model}: {original_filename}")
            
            # 将图片转为 base64
            base64_data = base64.b64encode(content).decode("utf-8")
            
            # 确定 MIME 类型
            mime_type = content_type
            if mime_type == "application/octet-stream":
                ext_mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", 
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }
                mime_type = ext_mime_map.get(file_ext, "image/png")
            
            # 确定 provider
            if model.startswith("dall-e"):
                provider = "dalle"
            elif model.startswith("gemini"):
                provider = "gemini"
            else:
                provider = "gpt"
            
            return {
                "file_id": f"{provider}-{uuid.uuid4().hex}",
                "filename": original_filename,
                "content": "",  # 图像模型不需要提取文本
                "image_base64": base64_data,
                "mime_type": mime_type,
                "provider": provider
            }
        
        # 对于 Kimi 模型或非图片文件，使用 Moonshot API
        if not MOONSHOT_API_KEY:
            raise HTTPException(status_code=500, detail="MOONSHOT_API_KEY not configured")
        
        # 使用安全的 ASCII 文件名和 MIME 类型，避免编码问题
        safe_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # 确保 content_type 是纯 ASCII
        safe_content_type = "application/octet-stream"
        if is_image:
            ext_mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            safe_content_type = ext_mime_map.get(file_ext, "image/png")
        elif file_ext == ".pdf":
            safe_content_type = "application/pdf"
        elif file_ext in [".doc", ".docx"]:
            safe_content_type = "application/msword"
        elif file_ext == ".txt":
            safe_content_type = "text/plain"
        
        async with httpx.AsyncClient() as client:
            # 1. Upload - 使用显式的 multipart 编码
            print(f"Uploading file to Moonshot: {original_filename}...")
            
            # 手动构建 multipart 数据，避免编码问题
            files_data = {
                "file": (safe_filename, content, safe_content_type),
                "purpose": (None, "file-extract"),
            }
            
            upload_res = await client.post(
                f"{MOONSHOT_BASE_URL}/files",
                headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"},
                files=files_data,
                timeout=60.0
            )
            
            if upload_res.status_code != 200:
                print(f"Upload failed: {upload_res.text}")
                raise HTTPException(status_code=upload_res.status_code, detail=f"Upload failed: {upload_res.text}")
            
            upload_json = upload_res.json()
            file_id = upload_json.get("id")
            if not file_id:
                 raise HTTPException(status_code=500, detail="Failed to get file_id from response")

            print(f"File uploaded, ID: {file_id}. Fetching content...")
            
            # 2. Extract with retry
            extracted_text = ""
            import asyncio
            
            for i in range(3): # Retry 3 times
                try:
                    content_res = await client.get(
                        f"{MOONSHOT_BASE_URL}/files/{file_id}/content",
                        headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"}
                    )
                    
                    if content_res.status_code == 200:
                        try:
                            json_content = content_res.json()
                            if "content" in json_content:
                                extracted_text = json_content["content"]
                                break 
                            elif "error" in json_content:
                                print(f"API Error in content: {json_content}")
                            else:
                                print(f"Unknown JSON structure: {json_content}")
                                extracted_text = str(json_content) 
                                break
                        except json.JSONDecodeError:
                            extracted_text = content_res.text
                            break
                    else:
                        print(f"Get content failed (attempt {i+1}): {content_res.status_code} - {content_res.text}")
                
                except Exception as e:
                    print(f"Exception during content fetch: {e}")

                await asyncio.sleep(1)
            
            if not extracted_text:
                extracted_text = "(未提取到文本内容，可能是图片不清晰或API处理超时)"
                print("Warning: Content extraction failed or empty.")

            return {
                "file_id": file_id,
                "filename": original_filename,
                "content": extracted_text,
                "provider": "moonshot"
            }

    except Exception as e:
        print(f"Error in upload_and_parse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def gemini_proxy_chat(messages: list, model: str):
    """
    使用代理 API (api.gptsapi.net) 调用 Gemini 模型
    使用 OpenAI 兼容格式
    """
    print(f"[Gemini Proxy] Using proxy API with model: {model}")
    
    # 流式响应生成器
    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{GPT_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GPT_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7,
                    },
                    timeout=120.0
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"[Gemini Proxy] Error {response.status_code}: {error_text[:200]}")
                    yield f"data: {{\"error\": \"{error_text[:100]}\"}}\n\n"
                    return
                
                # 直接转发 SSE 响应
                for line in response.text.split("\n"):
                    if line.strip():
                        yield f"{line}\n\n"
                        
        except Exception as e:
            print(f"[Gemini Proxy] Exception: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

async def google_gemini_chat(messages: list, model: str):
    """
    使用 Google 官方 Gemini API 进行文本对话
    根据官方文档: https://ai.google.dev/gemini-api/docs/text-generation
    """
    print(f"[Google Gemini] Using official API with model: {model}")
    
    # 转换消息格式：OpenAI -> Google Gemini
    gemini_contents = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            # 多模态内容
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # base64 图片
                        mime_type = image_url.split(";")[0].split(":")[1]
                        base64_data = image_url.split(",")[1]
                        parts.append({
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64_data
                            }
                        })
        
        if parts:
            gemini_contents.append({"role": role, "parts": parts})
    
    # 流式响应生成器
    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                # 使用流式 API
                response = await client.post(
                    f"{GOOGLE_GEMINI_BASE_URL}/models/{model}:streamGenerateContent?alt=sse&key={GOOGLE_API_KEY}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": gemini_contents,
                        "generationConfig": {
                            "temperature": 0.7,
                            "topP": 0.95,
                            "topK": 40,
                        }
                    },
                    timeout=120.0
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"[Google Gemini] Error {response.status_code}: {error_text[:200]}")
                    yield f"data: {{\"error\": \"{error_text[:100]}\"}}\n\n"
                    return
                
                # 解析 SSE 响应
                for line in response.text.split("\n"):
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            json_data = json.loads(data)
                            # 提取文本内容
                            candidates = json_data.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        # 转换为 OpenAI 格式
                                        openai_chunk = {
                                            "choices": [{
                                                "delta": {"content": part["text"]},
                                                "index": 0
                                            }]
                                        }
                                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
                
                yield "data: [DONE]\n\n"
                
        except Exception as e:
            print(f"[Google Gemini] Exception: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/generate-image")
async def generate_image(request: dict):
    """
    使用 DALL-E 生成或编辑图片，带重试机制
    - 如果提供 reference_image，则使用图像变体/编辑功能
    - 否则使用纯文本生成
    """
    if not GPT_API_KEY:
        raise HTTPException(status_code=500, detail="GPT_API_KEY not configured")
    
    prompt = request.get("prompt", "")
    model = request.get("model", "dall-e-3")
    size = request.get("size", "1024x1024")  # 支持: 1024x1024, 1024x1792, 1792x1024
    quality = request.get("quality", "standard")  # standard 或 hd
    reference_image = request.get("reference_image")  # base64 图片数据
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # 重试机制
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                print(f"[DALL-E] Attempt {attempt + 1}/{max_retries}: {prompt[:50]}...")
                
                # 如果有参考图片，使用图像编辑 API
                if reference_image:
                    print(f"[DALL-E] Using image edit mode with reference image")
                    
                    # 解码 base64 图片
                    image_bytes = base64.b64decode(reference_image)
                    
                    # 使用 DALL-E 2 的图像编辑 API（DALL-E 3 暂不支持编辑）
                    response = await client.post(
                        f"{GPT_BASE_URL}/images/edits",
                        headers={
                            "Authorization": f"Bearer {GPT_API_KEY}",
                        },
                        files={
                            "image": ("image.png", image_bytes, "image/png"),
                        },
                        data={
                            "prompt": prompt,
                            "n": 1,
                            "size": "1024x1024",  # 编辑 API 只支持 1024x1024
                            "response_format": "url"
                        },
                        timeout=180.0
                    )
                else:
                    # 纯文本生成
                    response = await client.post(
                        f"{GPT_BASE_URL}/images/generations",
                        headers={
                            "Authorization": f"Bearer {GPT_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "prompt": prompt,
                            "n": 1,
                            "size": size,
                            "quality": quality,
                            "response_format": "url"
                        },
                        timeout=180.0
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    image_url = result["data"][0]["url"]
                    revised_prompt = result["data"][0].get("revised_prompt", prompt)
                    
                    print(f"[DALL-E] Success!")
                    return {
                        "success": True,
                        "image_url": image_url,
                        "revised_prompt": revised_prompt,
                        "original_prompt": prompt,
                        "mode": "edit" if reference_image else "generate"
                    }
                else:
                    error_text = response.text
                    print(f"[DALL-E] Error {response.status_code}: {error_text}")
                    last_error = f"Image generation failed: {error_text}"
                    
                    # 如果是 500 错误，等待后重试
                    if response.status_code >= 500:
                        import asyncio
                        await asyncio.sleep(2 * (attempt + 1))  # 递增等待时间
                        continue
                    else:
                        # 非 500 错误直接返回
                        raise HTTPException(status_code=response.status_code, detail=last_error)
                        
        except httpx.TimeoutException:
            last_error = "图片生成超时，请稍后重试"
            print(f"[DALL-E] Timeout on attempt {attempt + 1}")
            continue
        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"[DALL-E] Exception on attempt {attempt + 1}: {e}")
            continue
    
    # 所有重试都失败
    raise HTTPException(status_code=500, detail=f"图片生成失败，请稍后重试。错误：{last_error}")

@app.post("/api/chat")
async def chat_proxy(request: dict):
    messages = request.get("messages", [])
    model = request.get("model", "kimi-k2-0905-preview")
    enable_image_gen = request.get("enable_image_generation", False)
    image_prompt = request.get("image_prompt", "")
    
    print(f"[DEBUG] Chat request - model: {model}, messages count: {len(messages)}")
    
    # Google Gemini 模型 (通过代理 API)
    google_gemini_models = [
        "gemini-3-pro-preview",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-05-20",
    ]
    if model in google_gemini_models:
        if not GPT_API_KEY:
            raise HTTPException(status_code=500, detail="GPT_API_KEY not configured for Gemini proxy")
        return await gemini_proxy_chat(messages, model)
    
    # Gemini 图像生成模型 (使用代理 API)
    if model.startswith("gemini"):
        if not GPT_API_KEY:
            raise HTTPException(status_code=500, detail="GPT_API_KEY not configured")
        return await gemini_image_generation(messages, model, image_prompt)
    
    # 根据模型名称选择对应的 API
    if model.startswith("gpt"):
        api_key = GPT_API_KEY
        base_url = GPT_BASE_URL
        if not api_key:
            raise HTTPException(status_code=500, detail="GPT_API_KEY not configured")
    else:
        api_key = MOONSHOT_API_KEY
        base_url = MOONSHOT_BASE_URL
        if not api_key:
            raise HTTPException(status_code=500, detail="MOONSHOT_API_KEY not configured")
    
    # GPT 模型启用图像生成：调用文本 + DALL-E 组合模式
    if model.startswith("gpt") and enable_image_gen:
        return await chat_with_image_generation(messages, model, api_key, base_url, image_prompt)
    
    # 标准流式聊天
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.3
    
    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", 
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"API Error ({model}): {response.status_code} - {error_text}")
                        yield f"data: Error: {response.status_code}\n\n"
                        return
                        
                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
        except Exception as e:
            import traceback
            print(f"[ERROR] Stream exception: {e}")
            traceback.print_exc()
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def chat_with_image_generation(messages: list, model: str, api_key: str, base_url: str, image_prompt: str):
    """
    GPT 文本回复 + 自动调用 DALL-E 3 生成图片
    """
    if not image_prompt:
        # 默认使用用户最后一条消息
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    image_prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            image_prompt = part.get("text", "")
                            break
                break
    
    async with httpx.AsyncClient() as client:
        # 1. 获取 GPT 文本回复（非流式）
        chat_payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if not model.startswith("gpt-5"):
            chat_payload["temperature"] = 0.3
        
        chat_response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=chat_payload,
            timeout=60.0
        )
        
        if chat_response.status_code != 200:
            raise HTTPException(status_code=chat_response.status_code, detail=chat_response.text)
        
        chat_result = chat_response.json()
        content = chat_result["choices"][0]["message"]["content"]
        
        response_data = {
            "content": content or "",
            "images": []
        }
        
        # 2. 调用 DALL-E 3 生成图片（带重试，使用 URL 格式）
        if image_prompt:
            import asyncio
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    print(f"[DALL-E] Attempt {attempt + 1}/{max_retries}: {image_prompt[:50]}...")
                    dalle_response = await client.post(
                        f"{base_url}/images/generations",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "dall-e-3",
                            "prompt": image_prompt,
                            "n": 1,
                            "size": "1024x1024",
                            "quality": "standard",
                            "response_format": "url"  # 使用 URL 格式，和直接调用保持一致
                        },
                        timeout=180.0
                    )
                    
                    if dalle_response.status_code == 200:
                        dalle_result = dalle_response.json()
                        if "data" in dalle_result and len(dalle_result["data"]) > 0:
                            data = dalle_result["data"][0]
                            image_url = data.get("url", "")
                            revised_prompt = data.get("revised_prompt", image_prompt)
                            
                            if image_url:
                                response_data["images"].append({
                                    "url": image_url,  # 返回 URL 而不是 base64
                                    "type": "url"
                                })
                                response_data["content"] += f"\n\n**图片提示词：** {revised_prompt}"
                                print(f"[DALL-E] Success!")
                                break
                    else:
                        print(f"[DALL-E] Error {dalle_response.status_code}: {dalle_response.text}")
                        if dalle_response.status_code >= 500 and attempt < max_retries - 1:
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        else:
                            response_data["content"] += "\n\n（图片生成失败，请稍后重试）"
                            break
                            
                except httpx.TimeoutException:
                    print(f"[DALL-E] Timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    response_data["content"] += "\n\n（图片生成超时，请稍后重试）"
                except Exception as e:
                    print(f"[DALL-E] Exception: {e}")
                    response_data["content"] += "\n\n（图片生成出错，请稍后重试）"
                    break
    
    return JSONResponse(content=response_data)


async def gemini_image_generation(messages: list, model: str, image_prompt: str, reference_image: str = None):
    """
    Gemini 图像生成
    - 优先使用 Google 官方 API（支持真正的图生图）
    - 降级到代理 API
    """
    # 从消息中提取用户最后一条文本和图片
    if not image_prompt or not reference_image:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    if not image_prompt:
                        image_prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text" and not image_prompt:
                            image_prompt = part.get("text", "")
                        elif part.get("type") == "image_url" and not reference_image:
                            image_url_data = part.get("image_url", {}).get("url", "")
                            if image_url_data.startswith("data:"):
                                reference_image = image_url_data.split(",", 1)[-1]
                break
    
    if not image_prompt:
        return JSONResponse(content={"content": "请提供图片描述", "images": []})
    
    import asyncio
    
    async with httpx.AsyncClient() as client:
        try:
            # ===== 优先使用 Google 官方 API =====
            if GOOGLE_API_KEY:
                # Gemini 图像生成模型
                gemini_model = "gemini-2.0-flash-exp-image-generation"
                print(f"[Gemini] Using Google Official API with model: {gemini_model}")
                
                # 如果有参考图片，增强 prompt 以尽量保持人物特征
                enhanced_prompt = image_prompt
                if reference_image:
                    enhanced_prompt = f"""Based on the reference image provided, {image_prompt}

IMPORTANT: Maintain the EXACT same person's facial features, face shape, eye shape, nose, lips, skin tone, and overall appearance from the reference image. The generated image should look like the SAME PERSON, just in a different style or setting."""
                
                parts_with_prompt = []
                if reference_image:
                    parts_with_prompt.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": reference_image
                        }
                    })
                parts_with_prompt.append({"text": enhanced_prompt})
                
                # 调用 Google 官方 Gemini API (Nano Banana)
                response = await client.post(
                    f"{GOOGLE_GEMINI_BASE_URL}/models/{gemini_model}:generateContent?key={GOOGLE_API_KEY}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": parts_with_prompt}],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"]
                        }
                    },
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[Gemini] Official API response received")
                    
                    response_data = {"content": "", "images": []}
                    
                    # 解析响应
                    candidates = result.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        for part in content.get("parts", []):
                            if "text" in part:
                                response_data["content"] += part["text"]
                            elif "inlineData" in part:
                                # 图片数据
                                inline_data = part["inlineData"]
                                mime_type = inline_data.get("mimeType", "image/png")
                                image_data = inline_data.get("data", "")
                                if image_data:
                                    response_data["images"].append({
                                        "url": f"data:{mime_type};base64,{image_data}",
                                        "type": "base64"
                                    })
                    
                    if response_data["images"]:
                        if not response_data["content"]:
                            response_data["content"] = "✨ 图片已生成！"
                        print(f"[Gemini] Success! Generated {len(response_data['images'])} image(s)")
                        return JSONResponse(content=response_data)
                    else:
                        print(f"[Gemini] No images in response, trying proxy API...")
                else:
                    print(f"[Gemini] Official API failed: {response.status_code} - {response.text[:200]}")
            
            # ===== 降级到代理 API =====
            if not GPT_API_KEY:
                return JSONResponse(content={"content": "API Key 未配置", "images": []})
            
            print(f"[Gemini] Falling back to proxy API...")
            
            # 如果有参考图片但官方 API 失败，使用分析+生成的方式
            final_prompt = image_prompt
            if reference_image:
                print(f"[Gemini] Analyzing reference image with vision model...")
                vision_response = await client.post(
                    f"{GPT_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GPT_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Analyze this image and create a detailed prompt for regenerating it with this modification: {image_prompt}\n\nOutput only the English prompt."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{reference_image}"}}
                            ]
                        }],
                        "max_tokens": 800
                    },
                    timeout=60.0
                )
                if vision_response.status_code == 200:
                    final_prompt = vision_response.json()["choices"][0]["message"]["content"]
                    print(f"[Gemini] Analyzed prompt: {final_prompt[:100]}...")
            
            # 提交到代理 API
            print(f"[Gemini] Submitting to proxy API: {final_prompt[:50]}...")
            submit_response = await client.post(
                f"{GEMINI_PROXY_BASE_URL}/google/{model}/text-to-image",
                headers={
                    "Authorization": f"Bearer {GPT_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "prompt": final_prompt,
                    "aspect_ratio": "1:1",
                    "output_format": "png"
                },
                timeout=60.0
            )
            
            if submit_response.status_code != 200:
                return JSONResponse(content={
                    "content": f"图片生成失败：{submit_response.text}",
                    "images": []
                })
            
            submit_result = submit_response.json()
            result_id = submit_result.get("data", {}).get("id")
            if not result_id:
                return JSONResponse(content={"content": "未获取到任务ID", "images": []})
            
            print(f"[Gemini] Task submitted, polling result_id: {result_id}")
            
            # 轮询结果
            for poll in range(60):
                await asyncio.sleep(3)
                result_response = await client.get(
                    f"{GEMINI_PROXY_BASE_URL}/predictions/{result_id}/result",
                    headers={"Authorization": f"Bearer {GPT_API_KEY}"},
                    timeout=30.0
                )
                
                if result_response.status_code != 200:
                    continue
                
                result_data = result_response.json()
                status = result_data.get("data", {}).get("status", "")
                
                if status in ["completed", "succeeded"]:
                    outputs = result_data.get("data", {}).get("outputs", [])
                    response_data = {"content": "✨ 图片已生成！", "images": []}
                    
                    for output in outputs:
                        if isinstance(output, str):
                            img_type = "url" if output.startswith("http") else "base64"
                            response_data["images"].append({"url": output, "type": img_type})
                        elif isinstance(output, dict):
                            url = output.get("url") or output.get("image")
                            if url:
                                img_type = "url" if url.startswith("http") else "base64"
                                response_data["images"].append({"url": url, "type": img_type})
                    
                    return JSONResponse(content=response_data)
                
                elif status in ["failed", "error"]:
                    return JSONResponse(content={
                        "content": f"图片生成失败：{result_data.get('data', {}).get('error', '未知错误')}",
                        "images": []
                    })
            
            return JSONResponse(content={"content": "图片生成超时", "images": []})
                
        except Exception as e:
            print(f"[Gemini] Exception: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(content={"content": f"图片生成出错：{str(e)}", "images": []})


# Static files (Frontend)
if os.path.exists("../frontend/dist"):
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys
    import logging
    
    # 抑制 Windows 上的 ConnectionResetError 警告
    if sys.platform == "win32":
        # 重写 stderr 来过滤特定错误信息
        class FilteredStderr:
            def __init__(self, original):
                self.original = original
                self.buffer = ""
                self.skip_until_empty = False
            
            def write(self, text):
                self.buffer += text
                # 当遇到换行时检查是否需要过滤
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    
                    # 检测错误块的开始
                    filter_keywords = [
                        "ConnectionResetError", "10054", "_call_connection_lost",
                        "_ProactorBasePipeTransport", "SHUT_RDWR", "socket.SHUT",
                        "proactor_events.py", "asyncio/events.py", "_context.run"
                    ]
                    
                    should_filter = any(kw in line for kw in filter_keywords)
                    
                    # 检测 Traceback 开始，如果后续有过滤关键词则跳过整个块
                    if "Traceback (most recent call last):" in line:
                        self.skip_until_empty = True
                        continue
                    
                    if self.skip_until_empty:
                        if line.strip() == "" or (not line.startswith(" ") and ":" not in line):
                            self.skip_until_empty = False
                        continue
                    
                    if not should_filter:
                        self.original.write(line + "\n")
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        sys.stderr = FilteredStderr(sys.stderr)
        
        # 设置异常处理器
        def silence_connection_reset(loop, context):
            exception = context.get("exception")
            if isinstance(exception, (ConnectionResetError, OSError)):
                return  # 静默忽略
            loop.default_exception_handler(context)
        
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(silence_connection_reset)
        asyncio.set_event_loop(loop)
    
    # Check for SSL files
    ssl_config = {}
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        print("Running in HTTPS mode with self-signed certificate.")
        ssl_config["ssl_certfile"] = "cert.pem"
        ssl_config["ssl_keyfile"] = "key.pem"
    else:
        print("Warning: Running in HTTP mode. Screen capture may not work on remote devices.")
        print("Run 'python gen_cert.py' to generate SSL certificates.")

    uvicorn.run(app, host="0.0.0.0", port=8000, **ssl_config)
