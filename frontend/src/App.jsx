import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import {
  Send, Plus, Globe, Cpu, BookOpen, Clock, Settings,
  MessageSquare, LayoutDashboard, Wrench, Moon, User,
  Paperclip, ShieldCheck, Zap, Activity, Bot, ChevronDown, CheckCheck,
  Search, Calculator, Code, FileText, Database, Trash2, Sprout,
  KeyRound, Sparkles, ChevronLeft, ChevronRight, Link2, Upload,
  AlertCircle, CheckCircle2, RefreshCw, X, Sun, Eye, Mic, ArrowRight,
  MonitorPlay, Languages, Scaling, Square, Copy, Edit2, Check, Pin
} from 'lucide-react'

const API_BASE = '/api'
const genSessionId = () => `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`

const ROUTE_META = {
  'general':  { label: 'General AI',        icon: Sparkles, color: '#D97706', bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  'web':      { label: 'Web Search',        icon: Globe,    color: '#0284C7', bg: 'bg-sky-50', text: 'text-sky-700', border: 'border-sky-200' },
  'finance':  { label: 'Stock & Cricket',   icon: Activity, color: '#EA580C', bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
  'farmer':   { label: 'Farmer Advisory',   icon: Sprout,   color: '#059669', bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  'rag':      { label: 'Document Analysis', icon: BookOpen, color: '#7C3AED', bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
}

const QUICK_PROMPTS = [
  {
    icon: Activity,
    title: 'Live Cricket & Scores',
    query: 'Get live cricket scores, match updates, player stats, and detailed commentary.',
    badge: 'CRICKET',
    color: '#D97706',
    bgColor: '#FEF3C7',
  },
  {
    icon: Globe,
    title: 'Real-time Web Search',
    query: 'Search the web in real-time and get the latest news, trends, and breakthroughs.',
    badge: 'WEB SEARCH',
    color: '#0284C7',
    bgColor: '#E0F2FE',
  },
  {
    icon: Sprout,
    title: 'Farmer Advisory',
    query: 'Get crop advice, mandi rates, fertilizer information, and government schemes.',
    badge: 'FARMER MODE',
    color: '#059669',
    bgColor: '#D1FAE5',
  },
  {
    icon: Cpu,
    title: 'Deep Research & Tech',
    query: 'In-depth research on any topic with verified sources, papers, and intelligent insights.',
    badge: 'RESEARCH',
    color: '#7C3AED',
    bgColor: '#EDE9FE',
  },
]

function formatRelativeTime(isoString) {
  if (!isoString) return ''
  try {
    const timeStr = isoString.endsWith('Z') || isoString.includes('+') ? isoString : isoString + 'Z';
    const dt = new Date(timeStr)
    const now = new Date()
    const diffMins = Math.floor((now - dt) / 60000)
    if (diffMins < 1) return 'just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return `${Math.floor(diffHours / 24)}d ago`
  } catch {
    return ''
  }
}

function SourceChips({ sources }) {
  if (!sources || sources.length === 0) return null
  const valid = sources.filter(s => s.url)
  if (valid.length === 0) return null

  return (
    <div className="mt-4 pt-3 border-t border-gray-100">
      <div className="flex flex-col gap-3">
        {valid.map((s, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[14px] font-bold text-gray-900 hover:text-[#B45309] hover:underline"
            >
              {s.title || new URL(s.url).hostname}
            </a>
            {s.snippet && <p className="text-[13px] text-gray-600 line-clamp-3 leading-relaxed">{s.snippet}</p>}
            <a href={s.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1.5 text-[11px] text-gray-500 hover:text-gray-800">
              <Globe size={11} /> {new URL(s.url).hostname} · Read article →
            </a>
          </div>
        ))}
      </div>
    </div>
  )
}

function ChatMessage({ msg, isStreaming, onEdit }) {
  const isUser = msg.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`flex flex-col mb-8 animate-fade-in group/message`}>
      {isUser ? (
        <div className="self-end max-w-[85%] sm:max-w-[75%] relative">
          <div className="bg-[#0A0A0A] dark:bg-purple-900/60 text-white dark:text-purple-50 px-5 py-3.5 rounded-2xl rounded-tr-sm shadow-sm dark:border dark:border-purple-500/30 text-[15px] leading-relaxed break-words">
            {msg.content}
          </div>
          <div className="flex items-center justify-end gap-3 mt-2 opacity-0 group-hover/message:opacity-100 transition-opacity pr-1">
            <button onClick={() => onEdit(msg.content)} className="flex items-center gap-1 text-[11px] font-bold text-gray-400 hover:text-gray-800 dark:hover:text-purple-300 transition-colors">
              <Edit2 size={12} /> Edit
            </button>
            <button onClick={handleCopy} className="flex items-center gap-1 text-[11px] font-bold text-gray-400 hover:text-gray-800 dark:hover:text-purple-300 transition-colors">
              {copied ? <Check size={12} className="text-emerald-500 dark:text-emerald-400" /> : <Copy size={12} />} {copied ? 'Copied' : 'Copy'}
            </button>
            <div className="flex items-center gap-1 text-[11px] text-gray-400 ml-1">
              <span>Sent</span>
              <CheckCheck size={12} className="text-yellow-500 dark:text-purple-400" />
            </div>
          </div>
        </div>
      ) : (
        <div className="self-start w-full bg-white dark:bg-[#1a1a24] border border-gray-200 dark:border-purple-800/40 rounded-3xl rounded-tl-sm shadow-sm relative group/ai">
          {/* Header */}
          <div className="flex items-center gap-2.5 px-6 py-3.5 border-b border-gray-100 dark:border-purple-800/40 bg-gray-50/50 dark:bg-[#151520] rounded-t-3xl rounded-tl-sm">
            <div className="w-8 h-8 rounded-lg bg-[#F5C518] dark:bg-purple-600 flex items-center justify-center shadow-sm shrink-0">
              <Bot size={16} className="text-black dark:text-white" />
            </div>
            <span className="text-[15px] font-bold text-gray-900 dark:text-purple-100">NexAi</span>
            {msg.route && ROUTE_META[msg.route] && (() => {
              const meta = ROUTE_META[msg.route]
              const IconComp = meta.icon || Sparkles
              return (
                <span className={`flex items-center gap-1.5 text-[11px] font-bold ml-auto px-3 py-1 rounded-full border ${meta.bg} dark:bg-gray-800 ${meta.text} dark:text-gray-300 ${meta.border} dark:border-gray-700`}>
                  <IconComp size={12} /> {meta.label}
                </span>
              )
            })()}
          </div>
          
          <div className="px-6 py-5 text-[15px] leading-relaxed text-gray-800 dark:text-gray-200">
            {msg.content ? (
              <div className="prose max-w-none break-words">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    a: ({ node, ...props }) => (
                      <a 
                        {...props} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-[#0284C7] font-semibold hover:underline hover:text-[#0369A1] transition-colors"
                      />
                    )
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="flex items-center gap-2 py-1 text-gray-400">
                <span className="w-2.5 h-2.5 rounded-full bg-[#F5C518] animate-ping"></span>
                <span className="text-[14px] font-medium text-gray-600">Thinking...</span>
              </div>
            )}

            {/* Sources */}
            {!isUser && msg.sources && <SourceChips sources={msg.sources} />}
          </div>
          
          {msg.content && !isStreaming && (
            <div className="px-6 py-3 border-t border-gray-100 dark:border-purple-800/40 bg-gray-50/50 dark:bg-[#151520] rounded-b-3xl flex items-center justify-end opacity-0 group-hover/ai:opacity-100 transition-opacity">
              <button onClick={handleCopy} className="flex items-center gap-1.5 text-[11px] font-bold text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors bg-white dark:bg-gray-800 px-3 py-1.5 rounded-full border border-gray-200 dark:border-gray-700 shadow-sm">
                {copied ? <Check size={12} className="text-emerald-500 dark:text-emerald-400" /> : <Copy size={12} />} {copied ? 'Copied' : 'Copy Response'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function EmptyState({ onSelectPrompt }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 px-6 text-center w-full">
      {/* Brand Hero */}
      <div className="mb-10 hero-logo">
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-3xl bg-[#0A0A0A] shadow-xl ring-4 ring-yellow-400/20 mb-5 relative group">
          <span className="text-[36px] font-black">
            <span className="text-[#F5C518]">N</span>
            <span className="text-white">x</span>
          </span>
          <div className="absolute inset-0 rounded-3xl border border-[#F5C518] opacity-0 group-hover:opacity-100 scale-105 transition-all duration-300"></div>
        </div>

        <h1 className="text-5xl sm:text-[64px] font-black tracking-tight text-gray-950 mb-3 leading-none">
          <span className="text-black">Nex</span>
          <span className="text-[#F5C518]">Ai</span>
        </h1>

        <p className="text-[15px] sm:text-[17px] text-gray-500 font-medium max-w-2xl mx-auto leading-relaxed mt-4">
          Your universal multi-agent assistant for web research, live cricket scores, finance, documents, and agriculture.
        </p>

        {/* Feature Tags */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
          {Object.entries(ROUTE_META).map(([key, meta]) => {
            const Icon = meta.icon
            return (
              <span key={key} className="flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-bold shadow-sm border border-gray-100 bg-white" style={{ color: meta.color }}>
                <Icon size={14} /> {meta.label}
              </span>
            )
          })}
        </div>
      </div>

      {/* Quick Prompt Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full max-w-[950px] text-left">
        {QUICK_PROMPTS.map((item, idx) => {
          const Icon = item.icon
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.query)}
              className="p-7 min-h-[140px] rounded-[24px] bg-white border border-gray-200 shadow-sm hover:shadow-md hover:border-yellow-400 transition-all group flex flex-col justify-start text-left"
            >
              <div className="flex items-center gap-4 w-full">
                <span
                  className="w-10 h-10 rounded-[14px] flex items-center justify-center shadow-sm shrink-0"
                  style={{ backgroundColor: item.bgColor, color: item.color }}
                >
                  <Icon size={18} />
                </span>
                <div className="flex-1">
                  <h3 className="font-bold text-gray-900 text-[15px] group-hover:text-black leading-tight">
                    {item.title}
                  </h3>
                </div>
                <span className="text-[10px] font-black uppercase tracking-widest text-gray-400 shrink-0">
                  {item.badge}
                </span>
              </div>
              <p className="text-[13px] text-gray-500 leading-relaxed font-medium mt-4">
                {item.query}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(380)
  const [isResizing, setIsResizing] = useState(false)
  
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(() => genSessionId())
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [statusLog, setStatusLog] = useState('')
  const [abortController, setAbortController] = useState(null)

  const [isDarkMode, setIsDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark'
  })

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [isDarkMode])

  const [farmerMode, setFarmerMode] = useState(false)
  const [agentModel, setAgentModel] = useState('openai/gpt-oss-120b')
  const [userGroqKey, setUserGroqKey] = useState('')
  const [activeUrl, setActiveUrl] = useState('')
  const [uploadedFileIds, setUploadedFileIds] = useState([])
  const [uploadingPdf, setUploadingPdf] = useState(false)

  const textareaRef = useRef(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  // Sidebar resize logic
  const startResizing = useCallback((e) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  const stopResizing = useCallback(() => {
    setIsResizing(false)
  }, [])

  const resize = useCallback((mouseMoveEvent) => {
    if (isResizing) {
      let newWidth = mouseMoveEvent.clientX
      if (newWidth < 280) newWidth = 280
      if (newWidth > 600) newWidth = 600
      setSidebarWidth(newWidth)
    }
  }, [isResizing])

  useEffect(() => {
    if (isResizing) {
      window.addEventListener("mousemove", resize)
      window.addEventListener("mouseup", stopResizing)
    }
    return () => {
      window.removeEventListener("mousemove", resize)
      window.removeEventListener("mouseup", stopResizing)
    }
  }, [isResizing, resize, stopResizing])

  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/chat/sessions?limit=30`)
      if (res.ok) {
        const data = await res.json()
        setSessions(Array.isArray(data) ? data : [])
      }
    } catch (e) {
      console.error('Failed to fetch sessions:', e)
    }
  }, [])

  useEffect(() => { loadSessions() }, [loadSessions])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, statusLog])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 250)}px`
    }
  }, [inputText])

  const handleSelectSession = async (sessionId) => {
    setActiveSessionId(sessionId)
    setStatusLog('')
    try {
      const res = await fetch(`${API_BASE}/chat/history/${sessionId}`)
      if (res.ok) {
        const history = await res.json()
        setMessages(history.map(m => ({ role: m.role, content: m.content })))
      }
    } catch (e) { console.error('Failed to fetch history:', e) }
  }

  const handleNewChat = () => {
    setActiveSessionId(genSessionId())
    setMessages([])
    setStatusLog('')
  }

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation()
    try {
      const res = await fetch(`${API_BASE}/chat/session/${sessionId}`, { method: 'DELETE' })
      if (res.ok) {
        if (activeSessionId === sessionId) handleNewChat()
        loadSessions()
      }
    } catch (e) { console.error('Failed to delete session:', e) }
  }

  const handleTogglePin = async (e, sessionId) => {
    e.stopPropagation()
    try {
      const res = await fetch(`${API_BASE}/chat/session/${sessionId}/pin`, { method: 'PUT' })
      if (res.ok) {
        loadSessions()
      }
    } catch (e) { console.error('Failed to toggle pin:', e) }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadingPdf(true)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/upload/pdf`, { method: 'POST', body: formData })
      if (res.ok) {
        const data = await res.json()
        if (data.file_id) setUploadedFileIds(prev => [...prev, data.file_id])
      }
    } catch (err) { console.error('PDF upload error:', err) }
    finally {
      setUploadingPdf(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleStop = () => {
    if (abortController) {
      abortController.abort()
      setAbortController(null)
      setStreaming(false)
      setStatusLog('')
    }
  }

  const handleEditMessage = (content) => {
    setInputText(content)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const handleSend = async (queryText) => {
    const textToSend = queryText || inputText.trim()
    if (!textToSend || streaming) return

    setInputText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    const newMsgs = [...messages, { role: 'user', content: textToSend }]
    setMessages(newMsgs)
    setStreaming(true)
    setStatusLog('Agent is thinking...')

    const controller = new AbortController()
    setAbortController(controller)

    const assistantIndex = newMsgs.length
    setMessages([...newMsgs, { role: 'assistant', content: '', route: 'general', sources: [] }])

    const params = new URLSearchParams({
      session_id: activeSessionId,
      message: textToSend,
      file_ids: uploadedFileIds.join(','),
      active_url: activeUrl.trim(),
      user_groq_key: userGroqKey.trim(),
      agent_model: agentModel,
      farmer_mode: farmerMode ? 'true' : 'false',
    })

    try {
      const response = await fetch(`${API_BASE}/chat/stream?${params.toString()}`, {
        signal: controller.signal
      })
      if (!response.ok) throw new Error('Failed to connect to stream')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = null
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.replace('event:', '').trim()
          } else if (line.startsWith('data:')) {
            const dataStr = line.replace('data:', '').trim()
            if (!dataStr) continue
            try {
              const data = JSON.parse(dataStr)
              if (currentEvent === 'log' || data.node) {
                if (data.message) setStatusLog(data.message)
              } else if (currentEvent === 'result' || data.answer) {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[assistantIndex] = {
                    role: 'assistant',
                    content: data.answer,
                    route: data.route || 'general',
                    sources: data.sources || [],
                  }
                  return updated
                })
                setStatusLog('')
              } else if (currentEvent === 'error' || data.error) {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[assistantIndex] = {
                    role: 'assistant',
                    content: `⚠️ **Error:** ${data.error || 'Something went wrong.'}`,
                    route: 'general',
                  }
                  return updated
                })
                setStatusLog('')
              }
            } catch (jsonErr) {}
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setMessages(prev => {
          const updated = [...prev]
          const currentContent = updated[assistantIndex].content
          updated[assistantIndex] = {
            ...updated[assistantIndex],
            content: currentContent + (currentContent ? '\n\n' : '') + '*(Generation stopped by user)*',
          }
          return updated
        })
      } else {
        setMessages(prev => {
          const updated = [...prev]
          updated[assistantIndex] = {
            role: 'assistant',
            content: '⚠️ **Backend Connection Error:** Make sure the FastAPI server is running.',
            route: 'general',
          }
          return updated
        })
      }
      setStatusLog('')
    } finally {
      setStreaming(false)
      setAbortController(null)
      loadSessions()
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-screen w-full bg-white dark:bg-[#05050A] text-[#0A0A0A] dark:text-gray-100 overflow-hidden font-sans transition-colors duration-300">
      
      {/* ─── LEFT SIDEBAR ─────────────────────────────────────────────────── */}
      <aside
        style={{ width: sidebarOpen ? `${sidebarWidth}px` : '0px' }}
        className={`relative ${isResizing ? '' : 'transition-[width] duration-300'} ease-in-out bg-[#FAFAFA] dark:bg-[#151520] border-r border-gray-200 dark:border-purple-800/40 flex flex-col shrink-0 overflow-hidden z-20`}
      >
        <div className="p-6 flex items-center justify-between border-b border-gray-200 dark:border-purple-800/40 bg-white dark:bg-[#111118]">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-[#0A0A0A] flex items-center justify-center shadow-sm">
              <span className="font-black text-2xl">
                <span className="text-[#F5C518]">N</span>
                <span className="text-white">x</span>
              </span>
            </div>
            <div>
              <h2 className="text-2xl font-black leading-none tracking-tight">
                <span className="text-black dark:text-white">Nex</span>
                <span className="text-[#F5C518]">Ai</span>
              </h2>
              <p className="text-[11px] font-bold text-gray-500 uppercase tracking-widest mt-1.5">Universal Agent</p>
            </div>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="p-2 border border-gray-200 rounded-xl hover:bg-gray-100 transition-colors">
            <ChevronLeft size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2.5 py-4 rounded-2xl bg-[#0A0A0A] dark:bg-purple-600 text-[#F5C518] dark:text-white border border-transparent hover:bg-black dark:hover:bg-purple-700 font-bold text-[16px] shadow-lg transition-all hover:scale-[1.02]"
          >
            <Plus size={20} /> New Chat
          </button>

          <div className="space-y-6">


            <div>
              <label className="text-[12px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2 mb-2.5">
                <KeyRound size={14} className="text-[#F5C518]" /> Groq API Key (Optional)
              </label>
              <div className="relative">
                <input
                  type="password"
                  placeholder="•••••••••••••••••••••"
                  value={userGroqKey}
                  onChange={(e) => setUserGroqKey(e.target.value)}
                  className="w-full px-4 py-3.5 rounded-xl bg-white border border-gray-200 text-[14px] font-bold text-gray-900 shadow-sm outline-none focus:border-[#F5C518] focus:ring-2 focus:ring-yellow-400/20 transition-all leading-normal h-12 pr-10"
                />
                <Eye size={16} className="absolute right-4 top-[14px] text-gray-400" />
              </div>
              <p className="text-[11px] text-gray-400 mt-2 font-medium">Leave blank to use built-in key</p>
              <div className="mt-1.5">
                {userGroqKey.trim() ? (
                  <span className="text-[11px] font-bold text-emerald-700 flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Custom key active</span>
                ) : (
                  <span className="text-[11px] font-bold text-amber-700 flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[#F5C518]"></span> Built-in key active</span>
                )}
              </div>
            </div>

            <div>
              <label className="text-[12px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2 mb-2.5">
                <Link2 size={14} className="text-[#F5C518]" /> URL Context
              </label>
              <input
                type="url"
                placeholder="https://example.com/article"
                value={activeUrl}
                onChange={(e) => setActiveUrl(e.target.value)}
                className="w-full px-4 py-3.5 rounded-xl bg-white border border-gray-200 text-[14px] font-medium text-gray-900 shadow-sm outline-none focus:border-[#F5C518] focus:ring-2 focus:ring-yellow-400/20 transition-all leading-normal h-12"
              />
            </div>
          </div>

          <div className="pt-6 border-t border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <span className="text-[12px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <Clock size={14} className="text-[#F5C518]" /> Chat History ({sessions.length})
              </span>
              <button onClick={loadSessions} className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"><RefreshCw size={14} className="text-gray-400 hover:text-black" /></button>
            </div>
            <div className="space-y-2 max-h-72 overflow-y-auto pr-3">
              {sessions.map(s => {
                const isActive = s.id === activeSessionId
                return (
                  <div key={s.id} onClick={() => handleSelectSession(s.id)} className={`group flex items-center justify-between p-3.5 rounded-xl cursor-pointer transition-colors ${isActive ? 'bg-[#FEFCE8] border border-[#FDE047] shadow-sm' : 'hover:bg-gray-100 border border-transparent'}`}>
                    <div className="flex items-center gap-3 truncate pr-3">
                      <MessageSquare size={16} className={isActive ? 'text-black' : 'text-gray-400'} />
                      <span className={`text-[14px] truncate ${isActive ? 'font-bold text-black' : 'font-semibold text-gray-600'}`}>{s.title || 'New Chat'}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-[10px] font-bold text-gray-400 group-hover:hidden">{formatRelativeTime(s.updated_at)}</span>
                      <button onClick={(e) => handleTogglePin(e, s.id)} className={`${s.is_pinned ? 'block' : 'hidden group-hover:block'} p-1 transition-colors ${s.is_pinned ? 'text-[#F5C518]' : 'text-gray-400 hover:text-gray-600'}`}><Pin size={14} className={s.is_pinned ? 'fill-current' : ''} /></button>
                      <button onClick={(e) => handleDeleteSession(e, s.id)} className="hidden group-hover:block p-1 text-gray-400 hover:text-red-500 transition-colors"><Trash2 size={14} /></button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
        
        <div className="p-5 border-t border-gray-200 dark:border-purple-800/40 bg-white dark:bg-[#151520] flex items-center justify-between text-[12px] font-bold text-gray-500 dark:text-gray-400 shrink-0">
          <span className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500 dark:bg-emerald-400"></span> Agent Online</span>
          <div className="flex items-center gap-4">
            <span>v2.0.0</span>
            {isDarkMode ? (
              <Sun size={16} className="cursor-pointer hover:text-white text-yellow-400 transition-all" onClick={() => setIsDarkMode(false)} />
            ) : (
              <Moon size={16} className="cursor-pointer hover:text-black transition-colors" onClick={() => setIsDarkMode(true)} />
            )}
            <Settings size={16} className="cursor-pointer hover:text-black dark:hover:text-white transition-colors" />
          </div>
        </div>

        {/* Drag Handle for Resizing */}
        {sidebarOpen && (
          <div
            className="absolute top-0 right-0 w-2 h-full cursor-col-resize hover:bg-[#F5C518] z-30 transition-colors opacity-0 hover:opacity-100"
            onMouseDown={startResizing}
          />
        )}
      </aside>

      {/* ─── MAIN CENTER AREA ─────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col h-screen bg-white dark:bg-[#111118] relative">
        <header className="h-16 px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-4">
            {!sidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} className="flex items-center gap-1 font-black text-[22px]">
                <span className="text-[#F5C518]">N</span><span className="text-black dark:text-white">x</span>
              </button>
            )}
          </div>
          
          <div className="flex items-center gap-4">

            <button onClick={() => setIsDarkMode(!isDarkMode)} className="w-10 h-10 rounded-full border border-gray-200 dark:border-purple-800/40 flex items-center justify-center hover:bg-gray-50 dark:hover:bg-purple-900/30 dark:text-gray-300 transition-colors shadow-sm">
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <div className="w-10 h-10 rounded-full bg-[#1A1A5C] dark:bg-purple-700 text-white flex items-center justify-center text-[13px] font-bold shadow-sm">
              AS
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto min-h-0 px-12 lg:px-24">
          <div className="max-w-[900px] mx-auto w-full pb-12 pt-12 flex flex-col">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-start">
                <EmptyState onSelectPrompt={handleSend} />
              </div>
            ) : (
              <div className="py-6">
                {messages.map((msg, idx) => (
                  <ChatMessage 
                    key={idx} 
                    msg={msg} 
                    isStreaming={streaming && idx === messages.length - 1 && msg.role === 'assistant'} 
                    onEdit={handleEditMessage} 
                  />
                ))}
                {statusLog && (
                  <div className="flex items-center gap-3 max-w-[80%] mx-auto bg-gray-50 border border-gray-200 rounded-full px-5 py-3 text-[14px] font-bold text-gray-600 shadow-sm animate-fade-in mb-6">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#F5C518] animate-ping"></span>
                    {statusLog}
                  </div>
                )}
                <div ref={messagesEndRef} className="h-32" />
              </div>
            )}
          </div>
        </div>

        {/* INPUT BOX */}
        <div className="shrink-0 w-full px-12 lg:px-24 pb-6 lg:pb-8 flex justify-center bg-white dark:bg-transparent">
          <div className="w-full max-w-[900px] shadow-2xl rounded-[28px] bg-white dark:bg-[#1a1a24] border-2 border-gray-200 dark:border-purple-800/40 focus-within:border-[#F5C518] dark:focus-within:border-purple-500 focus-within:ring-4 focus-within:ring-yellow-400/10 dark:focus-within:ring-purple-500/20 transition-all p-2 flex flex-col">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything — weather, news, cricket score, stocks, research..."
              disabled={streaming}
              className="w-full max-h-[300px] bg-transparent text-[15px] text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-600 placeholder:font-medium resize-none focus:outline-none px-6 py-3"
            />
            
            <div className="flex items-center justify-between px-5 pb-1 pt-1">
              <div className="flex items-center gap-3">
                <div className="relative group">
                  <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
                  <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-gray-100 dark:hover:bg-purple-900/30 text-[14px] font-bold text-gray-600 dark:text-gray-300 transition-colors">
                    <Paperclip size={16} /> Attach
                    {uploadingPdf && <span className="text-emerald-500 dark:text-emerald-400 text-[11px] ml-1">Uploading...</span>}
                    {uploadedFileIds.length > 0 && <span className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-emerald-400 ml-1"></span>}
                  </button>
                </div>
                
                <button onClick={() => setFarmerMode(!farmerMode)} className={`flex items-center gap-2 px-4 py-2 rounded-full text-[14px] font-bold transition-colors ${farmerMode ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 shadow-sm' : 'hover:bg-gray-100 dark:hover:bg-purple-900/30 text-gray-600 dark:text-gray-300'}`}>
                  <Sprout size={16} className={farmerMode ? 'text-emerald-500 dark:text-emerald-400' : ''} /> Farmer Mode 
                </button>
                
                <div className="relative group flex items-center">
                  <select 
                    value={agentModel}
                    onChange={(e) => setAgentModel(e.target.value)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    title="Select Agent Model"
                  >
                    <option value="openai/gpt-oss-120b">openai/gpt-oss-120b</option>
                    <option value="qwen/qwen3.6-27b">qwen/qwen3.6-27b</option>
                  </select>
                  <button className="flex items-center gap-2 px-4 py-2 rounded-full text-[14px] font-bold text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-purple-900/30 transition-colors relative">
                    <Cpu size={16} className="text-[#F5C518] dark:text-purple-400" />
                    Model <ChevronDown size={16} className="text-gray-400 dark:text-gray-500" />
                  </button>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <button className="w-11 h-11 rounded-full bg-white dark:bg-[#1a1a24] border border-gray-200 dark:border-purple-800/40 flex items-center justify-center hover:bg-gray-50 dark:hover:bg-purple-900/30 text-gray-600 dark:text-gray-400 transition-colors shadow-sm">
                  <Mic size={18} />
                </button>
                <button 
                  onClick={() => streaming ? handleStop() : handleSend()} 
                  disabled={(!inputText.trim() && !streaming)} 
                  className={`w-11 h-11 rounded-full flex items-center justify-center transition-all duration-300 ${streaming ? 'bg-red-500 dark:bg-rose-600 text-white shadow-lg hover:scale-105' : (inputText.trim() ? 'bg-black dark:bg-purple-600 text-[#F5C518] dark:text-white shadow-lg hover:scale-105' : 'bg-gray-100 dark:bg-[#1a1a24] text-gray-400 dark:text-gray-600 dark:border dark:border-purple-800/40')}`}
                >
                  {streaming ? <Square size={16} fill="currentColor" /> : <ArrowRight size={20} strokeWidth={3} />}
                </button>
              </div>
            </div>
          </div>
        </div>
        
        {messages.length === 0 && (
          <div className="absolute bottom-3 left-0 right-0 text-center pointer-events-none">
            <p className="text-[12px] font-bold text-gray-400 tracking-wide">
              <span className="text-[#F5C518]">Nex</span><span className="text-black">Ai</span> can provide realtime data, live scores, and multi-agent search.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
