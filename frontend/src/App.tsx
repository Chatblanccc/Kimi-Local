import { useState, useRef, useEffect } from 'react'
import { Send, Paperclip, Loader2, Image as ImageIcon, X, Sparkles, Copy, Check, ChevronDown, ChevronUp, Brain } from 'lucide-react'
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
  { id: 'kimi-k2-0905-preview', name: 'Kimi K2 Preview', description: '快速响应' },
  { id: 'kimi-k2-thinking', name: 'Kimi K2 思考', description: '深度思考' },
  { id: 'moonshot-v1-128k', name: 'Moonshot V1 128K', description: '长文本处理' },
] as const

type ModelId = typeof AVAILABLE_MODELS[number]['id']

interface Message {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string // 思考过程内容
  files?: { name: string, id: string, content?: string }[]
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
  const [currentFile, setCurrentFile] = useState<{ name: string, id: string, content?: string } | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelId>('kimi-k2-0905-preview')
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const modelMenuRef = useRef<HTMLDivElement>(null)

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
  }, [isUploading])

  const handleUpload = async (file: File) => {
    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

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
            content: data.content
        })
    } catch (error) {
        console.error("Upload error:", error)
        alert("上传失败，请重试")
    } finally {
        setIsUploading(false)
    }
  }

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
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map(m => {
              let content = m.content
              if (m.files && m.files.length > 0) {
                  m.files.forEach(f => {
                      if (f.content) {
                          content += `\n\n[文件: ${f.name}]\n${f.content}\n`
                      }
                  })
              }
              return { role: m.role, content: content }
          }),
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
      setMessages(prev => [...prev, { role: 'assistant', content: '出错了，请稍后再试。' }])
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
            {/* 月球图标 */}
            <div className="w-7 h-7 relative">
              <svg viewBox="0 0 36 36" className="w-full h-full">
                {/* 月球主体 - 渐变灰色 */}
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
                {/* 月球圆形 */}
                <circle cx="18" cy="18" r="16" fill="url(#moonGradient)" />
                {/* 月球环形边框 */}
                <circle cx="18" cy="18" r="16" fill="none" stroke="#a0a0a0" strokeWidth="0.5" />
                {/* 陨石坑 */}
                <circle cx="12" cy="10" r="3" fill="url(#craterGradient)" opacity="0.6" />
                <circle cx="22" cy="14" r="2" fill="url(#craterGradient)" opacity="0.5" />
                <circle cx="14" cy="20" r="2.5" fill="url(#craterGradient)" opacity="0.5" />
                <circle cx="24" cy="24" r="3.5" fill="url(#craterGradient)" opacity="0.4" />
                <circle cx="8" cy="18" r="1.5" fill="url(#craterGradient)" opacity="0.4" />
                <circle cx="20" cy="8" r="1.5" fill="url(#craterGradient)" opacity="0.3" />
                <circle cx="26" cy="18" r="1.8" fill="url(#craterGradient)" opacity="0.35" />
              </svg>
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
                className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-56 bg-white rounded-xl border border-gray-200 shadow-lg overflow-hidden z-50"
              >
                <div className="p-1">
                  {AVAILABLE_MODELS.map((model) => (
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
              placeholder={isUploading ? "正在解析文件..." : "发送消息..."}
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
            由 Moonshot AI 提供支持 • 仅供 JNUA 内部使用
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
