from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional
import json

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

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"

if not MOONSHOT_API_KEY:
    print("Warning: MOONSHOT_API_KEY not found in environment variables.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not MOONSHOT_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    try:
        # Read file content
        content = await file.read()
        
        files = {
            'file': (file.filename, content, file.content_type)
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MOONSHOT_BASE_URL}/files",
                headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"},
                files=files,
                data={"purpose": "file-extract"}
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
async def upload_and_parse(file: UploadFile = File(...)):
    if not MOONSHOT_API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    try:
        content = await file.read()
        files = {'file': (file.filename, content, file.content_type)}
        
        async with httpx.AsyncClient() as client:
            # 1. Upload
            print(f"Uploading file: {file.filename}...")
            upload_res = await client.post(
                f"{MOONSHOT_BASE_URL}/files",
                headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"},
                files=files,
                data={"purpose": "file-extract"}
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
                "filename": file.filename,
                "content": extracted_text
            }

    except Exception as e:
        print(f"Error in upload_and_parse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_proxy(request: dict):
    messages = request.get("messages", [])
    model = request.get("model", "kimi-k2-0905-preview")  # 从前端获取模型，默认为 kimi-k2-0905-preview
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }
    
    async def event_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", 
                f"{MOONSHOT_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {MOONSHOT_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    yield f"data: Error: {response.status_code}\n\n"
                    return
                    
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
