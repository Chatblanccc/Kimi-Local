import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 后端协议配置：设置环境变量 VITE_BACKEND_HTTPS=true 使用 HTTPS
const useHttps = process.env.VITE_BACKEND_HTTPS === 'true'
const backendUrl = useHttps ? 'https://localhost:8000' : 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
        secure: false,  // 忽略自签名证书（HTTPS 模式需要）
      }
    }
  }
})