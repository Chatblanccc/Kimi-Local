import path from "path"
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // 加载 .env 文件中的环境变量
  const env = loadEnv(mode, process.cwd(), '')
  const useHttps = env.VITE_BACKEND_HTTPS === 'true'
  const backendUrl = useHttps ? 'https://localhost:8000' : 'http://localhost:8000'
  
  console.log(`[Vite] Backend URL: ${backendUrl}`)  // 调试日志
  
  return {
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
  }
})