# Kimi Local 聊天助手

这是一个基于 React + FastAPI 的仿 Kimi 聊天网站，支持图片解析和屏幕截图。

## 部署说明

1. **配置 API Key**
   - 打开 `backend` 目录。
   - 复制 `.env.example` 为 `.env`。
   - 在 `.env` 中填入你的 Moonshot API Key: `MOONSHOT_API_KEY=sk-xxxxxxxx`。

2. **启动服务**
   - 双击根目录下的 `start.bat`。
   - 服务将启动在 `http://localhost:8000`。
   - 局域网访问请使用: `http://192.168.230.2:8000`。

## 功能特性
- **文本对话**: 支持 Markdown 渲染。
- **图片解析**: 上传图片或截图，自动调用 Kimi 解析内容并作为上下文发送。
- **屏幕截图**: 点击输入框上方的截图按钮，直接截取屏幕内容。

## 开发说明
- 前端: React + Vite + TailwindCSS + shadcn/ui
- 后端: FastAPI + Moonshot API
- 前端构建产物位于 `frontend/dist`，由后端静态托管。

### 开发环境启动
如果你需要二次开发，请按以下步骤启动调试服务：

**1. 启动后端 (端口 8000)**
```bash
cd backend
# 激活虚拟环境
venv\Scripts\activate
# 启动服务 (热重载)
uvicorn main:app --reload
```

**2. 启动前端 (端口 5173)**
```bash
cd frontend
npm run dev
```

