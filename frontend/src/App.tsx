import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Paperclip, Loader2, Image as ImageIcon, X, Sparkles, Copy, Check, ChevronDown, ChevronUp, Brain, Download, ZoomIn, ZoomOut } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScreenshotButton } from '@/components/ScreenshotButton'
import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

// 可用模型列表
const AVAILABLE_MODELS = [
  // Kimi 模型
  { id: 'kimi-k2-0905-preview', name: 'Kimi K2 Preview', description: '快速响应', provider: 'kimi' },
  { id: 'kimi-k2-thinking', name: 'Kimi K2 思考', description: '深度思考', provider: 'kimi' },
  { id: 'moonshot-v1-128k', name: 'Moonshot V1 128K', description: '长文本处理', provider: 'kimi' },
  // GPT 模型
  { id: 'gpt-5.1', name: 'GPT-5.1', description: '最新旗舰模型', provider: 'gpt' },
  { id: 'gpt-4o', name: 'GPT-4o', description: '多模态高效', provider: 'gpt' },
  // 图像生成模型
  { id: 'dall-e-3', name: 'DALL-E 3', description: 'OpenAI 图像生成', provider: 'dalle' },
  { id: 'gemini-3-pro-image-preview', name: 'Gemini 3 Pro', description: 'Google 图像生成', provider: 'gemini' },
] as const

type ModelId = typeof AVAILABLE_MODELS[number]['id']

interface FileData {
  name: string
  id: string
  content?: string
  image_base64?: string  // GPT 模型用的 base64 图片数据
  mime_type?: string     // 图片 MIME 类型
  provider?: 'gpt' | 'moonshot'
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string // 思考过程内容
  files?: FileData[]
  generatedImage?: {  // DALL-E 生成的图片
    url: string
    revisedPrompt?: string
  }
  isGeneratingImage?: boolean  // 是否正在生成图片
}

// 图片生成加载组件 - ChatGPT 风格
const ImageGeneratingBlock = () => {
  const [dots, setDots] = useState('')
  const [progress, setProgress] = useState(0)
  
  useEffect(() => {
    // 动态省略号
    const dotsInterval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.')
    }, 500)
    
    // 进度条动画 - 缓慢增长到 90%，留 10% 给完成时刻
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev
        // 越接近 90% 增长越慢
        const increment = Math.max(0.5, (90 - prev) / 50)
        return Math.min(prev + increment, 90)
      })
    }, 100)
    
    return () => {
      clearInterval(dotsInterval)
      clearInterval(progressInterval)
    }
  }, [])
  
  // 圆形进度条参数
  const size = 56
  const strokeWidth = 3
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const strokeDashoffset = circumference - (progress / 100) * circumference
  
  return (
    <div className="mt-4 mb-2">
      {/* Creating image 文字 */}
      <div className="text-sm text-gray-500 mb-3 font-medium">
        Creating image{dots}
      </div>
      
      {/* 图片占位框 - ChatGPT 风格 */}
      <div className="relative w-full max-w-md rounded-2xl overflow-hidden border border-gray-200 bg-gray-50">
        <div className="aspect-square w-full flex items-center justify-center">
          {/* 中心加载图标 + 圆形进度条 */}
          <div className="relative" style={{ width: size, height: size }}>
            {/* 背景圆环 */}
            <svg className="absolute inset-0 -rotate-90" width={size} height={size}>
              <circle
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="white"
                stroke="#e5e7eb"
                strokeWidth={strokeWidth}
              />
            </svg>
            
            {/* 进度圆环 */}
            <svg className="absolute inset-0 -rotate-90" width={size} height={size}>
              <circle
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="transparent"
                stroke="#000"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                style={{ transition: 'stroke-dashoffset 0.1s ease-out' }}
              />
            </svg>
            
            {/* 中心图标 */}
            <div className="absolute inset-0 flex items-center justify-center">
              <svg 
                className="w-6 h-6 text-gray-400" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="1.5"
              >
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M14 8h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// 图片预览灯箱组件
interface ImagePreviewProps {
  imageUrl: string
  onClose: () => void
}

const ImagePreview = ({ imageUrl, onClose }: ImagePreviewProps) => {
  const [scale, setScale] = useState(1)
  
  const handleZoomIn = () => setScale(prev => Math.min(prev + 0.25, 3))
  const handleZoomOut = () => setScale(prev => Math.max(prev - 0.25, 0.5))
  
  const handleDownload = async () => {
    try {
      const response = await fetch(imageUrl)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `generated-image-${Date.now()}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      // 如果下载失败，直接打开链接
      window.open(imageUrl, '_blank')
    }
  }
  
  // 点击背景关闭
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }
  
  // ESC 键关闭
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      onClick={handleBackdropClick}
    >
      {/* 顶部工具栏 */}
      <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
        <button
          onClick={handleZoomOut}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          title="缩小"
        >
          <ZoomOut className="w-5 h-5" />
        </button>
        <span className="text-white/70 text-sm min-w-[60px] text-center">
          {Math.round(scale * 100)}%
        </span>
        <button
          onClick={handleZoomIn}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          title="放大"
        >
          <ZoomIn className="w-5 h-5" />
        </button>
        <div className="w-px h-6 bg-white/20 mx-1" />
        <button
          onClick={handleDownload}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          title="下载图片"
        >
          <Download className="w-5 h-5" />
        </button>
        <button
          onClick={onClose}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          title="关闭"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      
      {/* 图片 */}
      <motion.img
        src={imageUrl}
        alt="Preview"
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl cursor-default"
        style={{ transform: `scale(${scale})` }}
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.2 }}
        onClick={(e) => e.stopPropagation()}
      />
      
      {/* 底部提示 */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/50 text-sm">
        按 ESC 或点击背景关闭 • 滚轮缩放
      </div>
    </motion.div>
  )
}

// 思考过程组件
const ThinkingBlock = ({ reasoning, isThinking }: { reasoning: string, isThinking: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(true)

  if (!reasoning && !isThinking) return null

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 transition-colors mb-2"
      >
        <Brain className={cn("w-4 h-4", isThinking && "animate-pulse text-purple-500")} />
        <span className="font-medium">{isThinking ? "正在思考..." : "思考过程"}</span>
        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="relative pl-4 border-l-2 border-purple-200 bg-gradient-to-r from-purple-50/50 to-transparent py-3 pr-3 rounded-r-lg">
              <div className="text-sm text-gray-600 leading-relaxed whitespace-pre-wrap">
                {reasoning || (
                  <span className="flex items-center gap-2 text-purple-500">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    思考中...
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// AI 回复底部的复制按钮组件
const CopyButton = ({ content }: { content: string }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex items-center gap-2 mt-3 pt-2 border-t border-gray-100">
      <button 
        onClick={handleCopy} 
        className="flex items-center gap-1.5 px-2 py-1 text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
        title="复制全文"
      >
        {copied ? (
          <>
            <Check className="w-3.5 h-3.5" />
            <span>已复制</span>
          </>
        ) : (
          <>
            <Copy className="w-3.5 h-3.5" />
            <span>复制</span>
          </>
        )}
      </button>
    </div>
  )
}

// 代码块组件 - 支持语法高亮和复制功能
const CodeBlock = ({ inline, className, children, ...props }: any) => {
  const [copied, setCopied] = useState(false)
  const match = /language-(\w+)/.exec(className || '')
  const codeString = String(children).replace(/\n$/, '')

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!inline && match) {
    return (
      <div className="relative group my-4">
        <div className="absolute right-2 top-2 z-10">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 text-gray-700 rounded transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3" />
                已复制
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                复制
              </>
            )}
          </button>
        </div>
        <SyntaxHighlighter
          style={oneLight}
          language={match[1]}
          PreTag="div"
          className="rounded-lg !bg-[#f7f7f8] !border !border-gray-200"
          {...props}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    )
  }

  return (
    <code className={cn("bg-gray-100 px-1.5 py-0.5 rounded text-sm text-red-600", className)} {...props}>
      {children}
    </code>
  )
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isThinking, setIsThinking] = useState(false) // 是否正在思考
  const [currentFile, setCurrentFile] = useState<FileData | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelId>('kimi-k2-0905-preview')
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  const [previewImage, setPreviewImage] = useState<string | null>(null) // 图片预览
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const modelMenuRef = useRef<HTMLDivElement>(null)

  const shouldAutoGenerateImage = useCallback((text: string) => {
    if (!text) return false
    const normalized = text.toLowerCase()
    const keywordList = [
      '生图', '画一', '画个', '画张', '画幅',
      '绘制', '生成图片', '生成一张', '设计一张', '做一张', '出一张',
      'picture', 'photo', 'image', 'draw', 'painting', 'illustration', 'render'
    ]
    if (keywordList.some(keyword => normalized.includes(keyword.toLowerCase()))) {
      return true
    }
    const regexList = [
      /画.*图/,
      /生成.*图/,
      /出.*图/,
      /做.*图/,
    ]
    return regexList.some(reg => reg.test(text))
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 点击外部关闭模型菜单
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setIsModelMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // 上传文件处理函数 - 使用 useCallback 确保 selectedModel 始终是最新值
  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', selectedModel)  // 传递当前选择的模型

    try {
        const response = await fetch('/api/upload_and_parse', {
            method: 'POST',
            body: formData
        })
        
        if (!response.ok) throw new Error('Upload failed')
        
        const data = await response.json()
        setCurrentFile({
            name: data.filename,
            id: data.file_id,
            content: data.content,
            image_base64: data.image_base64,
            mime_type: data.mime_type,
            provider: data.provider
        })
    } catch (error) {
        console.error("Upload error:", error)
        alert("上传失败，请重试")
    } finally {
        setIsUploading(false)
    }
  }, [selectedModel])

  // 粘贴图片处理
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            e.preventDefault() // Prevent default paste behavior
            const file = items[i].getAsFile()
            if (file) {
                handleUpload(file)
            }
            break
          }
        }
      }
    }

    document.addEventListener('paste', handlePaste)
    return () => {
      document.removeEventListener('paste', handlePaste)
    }
  }, [handleUpload])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleUpload(e.target.files[0])
    }
  }

  const handleScreenshot = (file: File) => {
      handleUpload(file)
  }

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if ((!inputValue.trim() && !currentFile) || isLoading) return

    let userMessageContent = inputValue
    const newMessage: Message = {
      role: 'user',
      content: userMessageContent,
      files: currentFile ? [currentFile] : undefined
    }
    
    const newMessages = [...messages, newMessage]
    setMessages(newMessages)
    setInputValue('')
    setCurrentFile(null)
    setIsLoading(true)

    try {
      // 检查是否是 DALL-E 图像生成模型
      if (selectedModel === 'dall-e-3') {
        // 先显示带加载动画的消息
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: '🎨 正在生成图片，请稍候...',
          isGeneratingImage: true 
        }])
        
        const response = await fetch('/api/generate-image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: userMessageContent,
            model: 'dall-e-3',
            size: '1024x1024',
            quality: 'standard'
          }),
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
          throw new Error(errorData.detail || 'Image generation failed')
        }

        const result = await response.json()
        
        // 更新消息，关闭加载动画，显示生成的图片
        setMessages(prev => {
          const newMsgs = [...prev]
          newMsgs[newMsgs.length - 1] = {
            role: 'assistant',
            content: result.revised_prompt !== result.original_prompt 
              ? `✨ 图片已生成！\n\n**优化后的提示词：** ${result.revised_prompt}` 
              : '✨ 图片已生成！',
            generatedImage: {
              url: result.image_url,
              revisedPrompt: result.revised_prompt
            },
            isGeneratingImage: false
          }
          return newMsgs
        })
        
        setIsLoading(false)
        return
      }
      
      // 检查是否是 Gemini 图像生成模型
      if (selectedModel.startsWith('gemini')) {
        // 先显示带加载动画的消息
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: '🎨 Gemini 正在生成图片...',
          isGeneratingImage: true 
        }])
        
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [{ role: 'user', content: userMessageContent }],
            model: selectedModel,
          }),
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
          throw new Error(errorData.detail || 'Image generation failed')
        }

        const result = await response.json()
        
        // 更新消息，关闭加载动画，显示生成的图片
        setMessages(prev => {
          const newMsgs = [...prev]
          const lastMsg: Message = {
            role: 'assistant',
            content: result.content || '✨ 图片已生成！',
            isGeneratingImage: false
          }
          
          // 如果有生成的图片
          if (result.images && result.images.length > 0) {
            lastMsg.generatedImage = {
              url: result.images[0].url,
              revisedPrompt: ''
            }
          }
          
          newMsgs[newMsgs.length - 1] = lastMsg
          return newMsgs
        })
        
        setIsLoading(false)
        return
      }

      // 根据模型类型构建不同格式的消息
      const isGptModel = selectedModel.startsWith('gpt')
      const wantsImage = isGptModel && shouldAutoGenerateImage(userMessageContent)
      
      const formattedMessages = newMessages.map(m => {
        // 如果是 GPT 模型且有图片，使用多模态格式
        if (isGptModel && m.files && m.files.some(f => f.image_base64)) {
          const contentParts: any[] = []
          
          // 添加文本内容
          if (m.content) {
            contentParts.push({ type: 'text', text: m.content })
          }
          
          // 添加图片
          m.files.forEach(f => {
            if (f.image_base64) {
              contentParts.push({
                type: 'image_url',
                image_url: {
                  url: `data:${f.mime_type || 'image/png'};base64,${f.image_base64}`
                }
              })
            }
          })
          
          return { role: m.role, content: contentParts }
        }
        
        // Kimi 模型或普通文本：使用文本格式
        let content = m.content
        if (m.files && m.files.length > 0) {
          m.files.forEach(f => {
            if (f.content) {
              content += `\n\n[文件: ${f.name}]\n${f.content}\n`
            }
          })
        }
        return { role: m.role, content: content }
      })
      
      // 如果是 GPT 模型且启用了图像生成，使用非流式请求
      if (wantsImage) {
        // 先显示带加载动画的消息
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: '🎨 正在为您生成图片...',
          isGeneratingImage: true 
        }])
        
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: formattedMessages,
            model: selectedModel,
            enable_image_generation: true,
            image_prompt: userMessageContent
          }),
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
          throw new Error(errorData.detail || 'Request failed')
        }

        const result = await response.json()
        
        // 更新消息，关闭加载状态
        setMessages(prev => {
          const newMsgs = [...prev]
          const lastMsg: Message = {
            role: 'assistant',
            content: result.content || '',
            isGeneratingImage: false
          }
          
          // 如果有生成的图片
          if (result.images && result.images.length > 0) {
            const img = result.images[0]
            // 支持 URL 和 base64 两种格式
            const imageUrl = img.type === 'url' 
              ? img.url 
              : `data:${img.type};base64,${img.base64}`
            
            lastMsg.generatedImage = {
              url: imageUrl,
              revisedPrompt: ''
            }
            if (!lastMsg.content) {
              lastMsg.content = '✨ 图片已生成！'
            }
          }
          
          newMsgs[newMsgs.length - 1] = lastMsg
          return newMsgs
        })
        
        setIsLoading(false)
        return
      }

      // 标准流式请求
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: formattedMessages,
          model: selectedModel,
          stream: true
        }),
      })

      if (!response.ok || !response.body) throw new Error('Network response was not ok')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      
      // 初始化 assistant 消息，包含空的思考内容
      setMessages(prev => [...prev, { role: 'assistant', content: '', reasoning: '' }])
      setIsThinking(true)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
             const data = line.slice(6)
             if (data === '[DONE]') continue
             try {
                 const json = JSON.parse(data)
                 const delta = json.choices[0]?.delta
                 
                 // 处理思考内容 (reasoning_content)
                 const reasoningContent = delta?.reasoning_content || ''
                 // 处理正常回复内容
                 const content = delta?.content || ''
                 
                 // 如果有正常内容输出，说明思考结束
                 if (content) {
                   setIsThinking(false)
                 }
                 
                 setMessages(prev => {
                     const last = prev[prev.length - 1]
                     if (last.role === 'assistant') {
                         return [...prev.slice(0, -1), { 
                           ...last, 
                           content: last.content + content,
                           reasoning: (last.reasoning || '') + reasoningContent
                         }]
                     }
                     return prev
                 })
             } catch (e) { }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => {
        // 如果最后一条是 assistant 的消息，更新它；否则添加新消息
        const last = prev[prev.length - 1]
        if (last && last.role === 'assistant') {
          return [...prev.slice(0, -1), { role: 'assistant', content: `出错了：${error instanceof Error ? error.message : '请稍后再试。'}` }]
        }
        return [...prev, { role: 'assistant', content: `出错了：${error instanceof Error ? error.message : '请稍后再试。'}` }]
      })
    } finally {
      setIsLoading(false)
      setIsThinking(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-[#fcfcfc] text-gray-900 font-sans">
      {/* Header */}
      <header className="sticky top-0 z-10 flex items-center justify-center p-2 bg-[#fcfcfc] border-b border-black/5">
        <div className="relative" ref={modelMenuRef}>
          <button
            onClick={() => setIsModelMenuOpen(!isModelMenuOpen)}
            className="flex items-center gap-2.5 p-2 rounded-md hover:bg-gray-100 transition-colors cursor-pointer"
          >
            {/* 根据模型显示不同图标 */}
            <div className="w-7 h-7 relative">
              {selectedModel === 'dall-e-3' ? (
                /* DALL-E 图像生成图标 */
                <svg viewBox="0 0 24 24" className="w-full h-full" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <defs>
                    <linearGradient id="dalleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#ec4899" />
                      <stop offset="100%" stopColor="#8b5cf6" />
                    </linearGradient>
                  </defs>
                  <rect x="3" y="3" width="18" height="18" rx="3" stroke="url(#dalleGradient)" />
                  <circle cx="8.5" cy="8.5" r="1.5" fill="url(#dalleGradient)" stroke="none" />
                  <path d="M21 15l-5-5L5 21" stroke="url(#dalleGradient)" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M14 10l3.5-3.5" stroke="url(#dalleGradient)" strokeLinecap="round" />
                  <path d="M3 15l4-4" stroke="url(#dalleGradient)" strokeLinecap="round" />
                </svg>
              ) : selectedModel.startsWith('gemini') ? (
                /* Gemini 官方图标 - 四角星渐变 */
                <svg viewBox="0 0 28 28" className="w-full h-full" fill="none">
                  <defs>
                    <linearGradient id="geminiGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#1BA1E3" />
                      <stop offset="25%" stopColor="#5489D6" />
                      <stop offset="50%" stopColor="#9B72CB" />
                      <stop offset="75%" stopColor="#D96570" />
                      <stop offset="100%" stopColor="#F49C46" />
                    </linearGradient>
                  </defs>
                  <path d="M14 0C14 7.732 7.732 14 0 14C7.732 14 14 20.268 14 28C14 20.268 20.268 14 28 14C20.268 14 14 7.732 14 0Z" fill="url(#geminiGradient)" />
                </svg>
              ) : selectedModel.startsWith('gpt') ? (
                /* OpenAI 官方图标 */
                <svg viewBox="0 0 24 24" className="w-full h-full text-black" fill="currentColor">
                  <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364l2.0201-1.1685a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" />
                </svg>
              ) : (
                /* 月球图标 (Kimi) */
                <svg viewBox="0 0 36 36" className="w-full h-full">
                  <defs>
                    <radialGradient id="moonGradient" cx="30%" cy="30%" r="70%">
                      <stop offset="0%" stopColor="#e8e8e8" />
                      <stop offset="50%" stopColor="#c4c4c4" />
                      <stop offset="100%" stopColor="#9a9a9a" />
                    </radialGradient>
                    <radialGradient id="craterGradient" cx="40%" cy="40%" r="60%">
                      <stop offset="0%" stopColor="#b0b0b0" />
                      <stop offset="100%" stopColor="#888888" />
                    </radialGradient>
                  </defs>
                  <circle cx="18" cy="18" r="16" fill="url(#moonGradient)" />
                  <circle cx="18" cy="18" r="16" fill="none" stroke="#a0a0a0" strokeWidth="0.5" />
                  <circle cx="12" cy="10" r="3" fill="url(#craterGradient)" opacity="0.6" />
                  <circle cx="22" cy="14" r="2" fill="url(#craterGradient)" opacity="0.5" />
                  <circle cx="14" cy="20" r="2.5" fill="url(#craterGradient)" opacity="0.5" />
                  <circle cx="24" cy="24" r="3.5" fill="url(#craterGradient)" opacity="0.4" />
                  <circle cx="8" cy="18" r="1.5" fill="url(#craterGradient)" opacity="0.4" />
                  <circle cx="20" cy="8" r="1.5" fill="url(#craterGradient)" opacity="0.3" />
                  <circle cx="26" cy="18" r="1.8" fill="url(#craterGradient)" opacity="0.35" />
                </svg>
              )}
            </div>
            <span className="font-semibold text-sm text-gray-700 tracking-wide">
              {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name || 'Jnua AI 助手'}
            </span>
            <ChevronDown className={cn(
              "w-4 h-4 text-gray-500 transition-transform duration-200",
              isModelMenuOpen && "rotate-180"
            )} />
          </button>

          {/* 模型选择下拉菜单 */}
          <AnimatePresence>
            {isModelMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.15 }}
                className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 bg-white rounded-xl border border-gray-200 shadow-lg overflow-hidden z-50 max-h-[70vh] overflow-y-auto"
              >
                <div className="p-1">
                  {/* Kimi 模型组 */}
                  <div className="px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">Kimi</div>
                  {AVAILABLE_MODELS.filter(m => m.provider === 'kimi').map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model.id)
                        setIsModelMenuOpen(false)
                      }}
                      className={cn(
                        "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors",
                        selectedModel === model.id 
                          ? "bg-blue-50 text-blue-700" 
                          : "hover:bg-gray-50 text-gray-700"
                      )}
                    >
                      <div className="flex-1">
                        <div className="font-medium text-sm">{model.name}</div>
                        <div className="text-xs text-gray-400">{model.description}</div>
                      </div>
                      {selectedModel === model.id && (
                        <Check className="w-4 h-4 text-blue-600" />
                      )}
                    </button>
                  ))}
                  
                  {/* 分隔线 */}
                  <div className="my-1 border-t border-gray-100" />
                  
                  {/* GPT 模型组 */}
                  <div className="px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">OpenAI</div>
                  {AVAILABLE_MODELS.filter(m => m.provider === 'gpt').map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model.id)
                        setIsModelMenuOpen(false)
                      }}
                      className={cn(
                        "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors",
                        selectedModel === model.id 
                          ? "bg-emerald-50 text-emerald-700" 
                          : "hover:bg-gray-50 text-gray-700"
                      )}
                    >
                      <div className="flex-1">
                        <div className="font-medium text-sm">{model.name}</div>
                        <div className="text-xs text-gray-400">{model.description}</div>
                      </div>
                      {selectedModel === model.id && (
                        <Check className="w-4 h-4 text-emerald-600" />
                      )}
                    </button>
                  ))}
                  
                  {/* 分隔线 */}
                  <div className="my-1 border-t border-gray-100" />
                  
                  {/* 图像生成模型组 */}
                  <div className="px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">🎨 图像生成</div>
                  {AVAILABLE_MODELS.filter(m => m.provider === 'dalle' || m.provider === 'gemini').map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model.id)
                        setIsModelMenuOpen(false)
                      }}
                      className={cn(
                        "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors",
                        selectedModel === model.id 
                          ? model.provider === 'gemini' ? "bg-blue-50 text-blue-700" : "bg-pink-50 text-pink-700"
                          : "hover:bg-gray-50 text-gray-700"
                      )}
                    >
                      <div className="flex-1">
                        <div className="font-medium text-sm">{model.name}</div>
                        <div className="text-xs text-gray-400">{model.description}</div>
                      </div>
                      {selectedModel === model.id && (
                        <Check className={cn("w-4 h-4", model.provider === 'gemini' ? "text-blue-600" : "text-pink-600")} />
                      )}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto scrollbar-hide bg-white">
        <div className="flex flex-col min-h-full pb-32 max-w-3xl mx-auto px-4 pt-8">
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 animate-in fade-in duration-500">
              <div className="bg-white p-4 rounded-full mb-6 shadow-sm border border-gray-100">
                <Sparkles className="w-12 h-12 text-blue-600" />
              </div>
              <h2 className="text-2xl font-semibold mb-2 text-gray-800">今天有什么可以帮您？</h2>
              <p className="text-gray-500 max-w-md">
                我可以帮您解析图片、回答问题，以及协助您的日常工作。
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={cn(
                  "flex w-full mb-6",
                  msg.role === 'user' ? "justify-end" : "justify-start"
                )}
              >
                <div className={cn(
                  "relative max-w-[90%] md:max-w-[85%] text-base",
                  msg.role === 'user' 
                    ? "bg-[#f4f4f4] rounded-2xl px-5 py-3 text-gray-900" 
                    : "bg-transparent px-0 text-gray-900 w-full"
                )}>
                    {msg.files && msg.files.length > 0 && (
                        <div className="mb-3 flex flex-wrap gap-2">
                            {msg.files.map((f, i) => (
                                <div key={i} className="flex items-center gap-2 bg-white p-2 rounded-md text-xs text-gray-600 border border-black/10 shadow-sm">
                                    <Paperclip className="w-3 h-3" />
                                    <span className="truncate max-w-[200px]">{f.name}</span>
                                </div>
                            ))}
                        </div>
                    )}
                    
                    {/* 思考过程显示 - 仅对 AI 消息且有思考内容时显示 */}
                    {msg.role === 'assistant' && (msg.reasoning || (isThinking && idx === messages.length - 1)) && (
                        <ThinkingBlock 
                          reasoning={msg.reasoning || ''} 
                          isThinking={isThinking && idx === messages.length - 1 && !msg.content} 
                        />
                    )}
                    
                    <div className="prose max-w-none leading-7 text-gray-800 prose-headings:text-gray-900 prose-strong:text-gray-900 prose-code:text-red-600 prose-pre:bg-gray-100 prose-pre:text-gray-900 prose-pre:border prose-pre:border-gray-200">
                      <ReactMarkdown
                        components={{
                          code: CodeBlock
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    
                    {/* 图片生成加载动画 */}
                    {msg.isGeneratingImage && (
                      <ImageGeneratingBlock />
                    )}
                    
                    {/* 显示生成的图片 */}
                    {msg.generatedImage && !msg.isGeneratingImage && (
                      <motion.div 
                        className="mt-4 w-full max-w-md"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div 
                          className="relative group aspect-square rounded-2xl overflow-hidden border border-gray-200 shadow-sm bg-gray-50 cursor-pointer"
                          onClick={() => setPreviewImage(msg.generatedImage!.url)}
                        >
                          <img 
                            src={msg.generatedImage.url} 
                            alt={msg.generatedImage.revisedPrompt || "Generated image"}
                            className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
                            loading="lazy"
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                            <div className="px-4 py-2 bg-white/95 text-gray-700 text-sm font-medium rounded-lg shadow-md">
                              点击查看大图
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                    
                    {msg.role === 'assistant' && !isLoading && msg.content && (
                        <CopyButton content={msg.content} />
                    )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} className="h-8" />
        </div>
      </main>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#fcfcfc] via-[#fcfcfc] to-transparent pt-10 pb-6">
        <div className="max-w-3xl mx-auto px-4">
          {currentFile && (
              <div className="mx-auto mb-2 max-w-3xl px-4">
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="inline-flex items-center gap-2 bg-white border border-gray-200 p-2 rounded-md text-sm text-gray-700 shadow-sm"
                >
                    <Paperclip className="w-4 h-4 text-gray-500" />
                    <span className="max-w-[200px] truncate">{currentFile.name}</span>
                    <button onClick={() => setCurrentFile(null)} className="ml-2 hover:text-red-500 text-gray-400 transition-colors">
                        <X className="w-4 h-4" />
                    </button>
                </motion.div>
              </div>
          )}

          <div className="relative flex items-center w-full p-3 bg-white rounded-xl border border-black/10 shadow-md overflow-hidden ring-offset-2 focus-within:ring-2 ring-blue-500/50">
             <Input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                onChange={handleFileChange}
                accept="image/*,.pdf,.doc,.docx,.txt" 
            />
            
            <Button 
                type="button" 
                variant="ghost" 
                size="icon"
                className="text-gray-500 hover:text-gray-700 hover:bg-gray-100 mr-1"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading || isLoading}
                title="上传文件"
            >
                <ImageIcon className="w-5 h-5" />
            </Button>

            <div className="mr-2">
                <ScreenshotButton onScreenshot={handleScreenshot} disabled={isUploading || isLoading} />
            </div>

            <input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                isUploading ? "正在解析文件..." : 
                selectedModel === 'dall-e-3' || selectedModel.startsWith('gemini') ? "描述你想生成的图片..." : 
                "发送消息..."
              }
              disabled={isUploading || isLoading}
              className="flex-1 bg-transparent border-0 outline-none text-gray-800 placeholder-gray-400 h-[24px] py-0"
              onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit()
                  }
              }}
            />
            
            <Button 
                type="submit" 
                onClick={() => handleSubmit()}
                disabled={(!inputValue.trim() && !currentFile) || isLoading || isUploading}
                className={cn(
                    "ml-2 transition-all duration-200",
                    inputValue.trim() || currentFile ? "bg-blue-600 hover:bg-blue-700 text-white" : "bg-gray-100 text-gray-400 cursor-not-allowed"
                )}
                size="icon"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </div>
          <div className="text-center text-xs text-gray-400 mt-2">
            由 Moonshot AI / OpenAI / Gemini 提供支持 • 仅供 JNUA 内部使用
          </div>
        </div>
      </div>
      
      {/* 图片预览灯箱 */}
      <AnimatePresence>
        {previewImage && (
          <ImagePreview 
            imageUrl={previewImage} 
            onClose={() => setPreviewImage(null)} 
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
