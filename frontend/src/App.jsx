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
  MonitorPlay, Languages, Scaling, Square, Copy, Edit2, Check, Pin, ExternalLink
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'
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
  const seen = new Set()
  const valid = sources.filter(s => {
    if (!s.url || seen.has(s.url)) return false
    seen.add(s.url)
    return true
  })
  if (valid.length === 0) return null

  return (
    <div className="mt-3.5 pt-3 border-t border-gray-100 dark:border-gray-800/80">
      <div className="flex items-center gap-1.5 mb-2 text-[11px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-400">
        <Globe size={12} className="text-sky-500" />
        <span>Verified Sources & Citations</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {valid.slice(0, 6).map((s, i) => {
          let host = ''
          try { host = new URL(s.url).hostname.replace(/^www\./, '') } catch { host = 'Source' }
          const label = s.source || host
          return (
            <a
              key={i}
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gray-50 dark:bg-[#151520] hover:bg-gray-100 dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700/60 text-[12px] font-semibold text-gray-700 dark:text-gray-300 transition-all hover:scale-[1.02] shadow-2xs group/chip"
              title={s.title || s.url}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-sky-500 shrink-0"></span>
              <span className="truncate max-w-[150px]">{label}</span>
              <ExternalLink size={10} className="text-gray-400 group-hover/chip:text-sky-500 shrink-0" />
            </a>
          )
        })}
      </div>
    </div>
  )
}

function getNodeMeta(nodeName) {
  const n = (nodeName || '').toLowerCase()
  if (n.includes('router')) {
    return {
      label: 'Router Node',
      icon: Sparkles,
      color: 'text-amber-600 dark:text-amber-400',
      dotBg: 'bg-amber-500',
      badgeBg: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800/40',
    }
  }
  if (n.includes('omni') || n.includes('agent')) {
    return {
      label: 'Omni-Agent Node',
      icon: Cpu,
      color: 'text-indigo-600 dark:text-purple-400',
      dotBg: 'bg-indigo-500 dark:bg-purple-500',
      badgeBg: 'bg-indigo-50 dark:bg-purple-950/40 text-indigo-700 dark:text-purple-300 border-indigo-200 dark:border-purple-800/40',
    }
  }
  if (n.includes('tool')) {
    return {
      label: 'Tools Execution Node',
      icon: Wrench,
      color: 'text-sky-600 dark:text-sky-400',
      dotBg: 'bg-sky-500',
      badgeBg: 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800/40',
    }
  }
  if (n.includes('retrieval') || n.includes('data')) {
    return {
      label: 'Data Retrieval Node',
      icon: Database,
      color: 'text-emerald-600 dark:text-emerald-400',
      dotBg: 'bg-emerald-500',
      badgeBg: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800/40',
    }
  }
  if (n.includes('synthesis')) {
    return {
      label: 'Synthesis Node',
      icon: Zap,
      color: 'text-yellow-600 dark:text-yellow-400',
      dotBg: 'bg-yellow-500',
      badgeBg: 'bg-yellow-50 dark:bg-yellow-950/40 text-yellow-700 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800/40',
    }
  }
  if (n.includes('farmer')) {
    return {
      label: 'Farmer Node',
      icon: Sprout,
      color: 'text-emerald-600 dark:text-emerald-400',
      dotBg: 'bg-emerald-500',
      badgeBg: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800/40',
    }
  }
  if (n.includes('completion') || n.includes('complete')) {
    return {
      label: 'Completion Node',
      icon: CheckCircle2,
      color: 'text-emerald-600 dark:text-emerald-400',
      dotBg: 'bg-emerald-500',
      badgeBg: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800/40',
    }
  }
  return {
    label: nodeName || 'Agent Step',
    icon: Activity,
    color: 'text-gray-600 dark:text-gray-400',
    dotBg: 'bg-gray-400',
    badgeBg: 'bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700',
  }
}

function ExecutionFlowTimeline({ steps, isStreaming }) {
  if (!steps || steps.length === 0) return null

  return (
    <div className="flex flex-col space-y-0 relative py-1">
      {steps.map((step, idx) => {
        const isLast = idx === steps.length - 1
        const rawNode = step.node || 'Agent'
        const meta = getNodeMeta(rawNode)
        const Icon = meta.icon
        const msgText = step.msg || step.message || ''

        const isTool = msgText.includes('tool_') || msgText.includes('Calling:') || rawNode.toLowerCase().includes('tool')

        return (
          <div key={idx} className="flex items-start gap-3.5 relative group animate-fade-in">
            {/* Step Indicator */}
            <div className="flex flex-col items-center self-stretch shrink-0">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center ${meta.dotBg} text-white shadow-xs z-10 ${isStreaming && isLast ? 'animate-pulse ring-4 ring-yellow-400/30' : ''}`}>
                <Icon size={12} strokeWidth={2.5} />
              </div>
              {!isLast && (
                <div className="w-0.5 flex-1 bg-gray-200 dark:bg-purple-900/60 my-1 min-h-[22px]"></div>
              )}
            </div>

            {/* Step Content */}
            <div className={`flex flex-col pb-3.5 flex-1 min-w-0 ${isLast ? 'pb-1' : ''}`}>
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-[11px] font-black uppercase tracking-wider px-2 py-0.5 rounded-md border ${meta.badgeBg}`}>
                  {meta.label}
                </span>
                {isStreaming && isLast && (
                  <span className="flex items-center gap-1 text-[10px] font-bold text-yellow-600 dark:text-yellow-400 animate-pulse">
                    <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-ping"></span>
                    Executing...
                  </span>
                )}
              </div>
              
              <div className="mt-1 text-[13px] leading-relaxed text-gray-700 dark:text-gray-300 break-words">
                {isTool ? (
                  <code className="bg-gray-100 dark:bg-gray-800 text-sky-700 dark:text-sky-300 px-2 py-0.5 rounded font-mono text-[12px] font-semibold inline-block">
                    {msgText}
                  </code>
                ) : (
                  <span>{msgText}</span>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ChatMessage({ msg, isStreaming, currentLogs, onEdit }) {
  const isUser = msg.role === 'user'
  const [copied, setCopied] = useState(false)
  const [showFlow, setShowFlow] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const flowSteps = msg.execution_flow || (isStreaming ? currentLogs : [])

  return (
    <div className={`flex flex-col mb-8 animate-fade-in group/message`}>
      {isUser ? (
        <div className="self-end max-w-[85%] sm:max-w-[75%] relative flex flex-col">
          {/* Attached Files */}
          {msg.files && msg.files.length > 0 && (
            <div className="flex flex-wrap justify-end gap-2 mb-2">
              {msg.files.map((f, i) => (
                <div key={i} className="flex items-center gap-2.5 bg-gray-100 dark:bg-[#1a1a24] text-gray-800 dark:text-gray-200 px-3 py-2 rounded-xl shadow-sm border border-gray-200 dark:border-purple-800/40 text-[13px] font-bold">
                  <div className="w-8 h-8 rounded-lg bg-[#F5C518] dark:bg-purple-600 flex items-center justify-center shrink-0 shadow-sm">
                    <FileText size={16} className="text-black dark:text-white" />
                  </div>
                  <span className="truncate max-w-[180px]">{f.name}</span>
                </div>
              ))}
            </div>
          )}
          <div className="bg-[#0A0A0A] dark:bg-purple-900/60 text-white dark:text-purple-50 px-5 py-3.5 rounded-2xl rounded-tr-sm shadow-sm dark:border dark:border-purple-500/30 text-[15px] leading-relaxed break-words self-end w-fit ml-auto">
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
                    a: ({ node, children, href, ...props }) => {
                      const text = String(children || '')
                      const isActionBtn = text.includes('→') || text.includes('Read') || text.includes('Source')
                      if (isActionBtn) {
                        return (
                          <a 
                            {...props} 
                            href={href}
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="inline-flex items-center gap-1.5 px-3 py-1 my-1.5 rounded-lg bg-yellow-50 dark:bg-purple-950/50 text-yellow-900 dark:text-purple-200 hover:bg-yellow-100 dark:hover:bg-purple-900/80 border border-yellow-200 dark:border-purple-800/60 font-bold text-[12px] transition-all hover:scale-[1.02] shadow-2xs no-underline group/btn cursor-pointer"
                          >
                            <span>{children}</span>
                            <ExternalLink size={11} className="text-yellow-600 dark:text-purple-400 group-hover/btn:translate-x-0.5 transition-transform" />
                          </a>
                        )
                      }
                      return (
                        <a 
                          {...props} 
                          href={href}
                          target="_blank" 
                          rel="noopener noreferrer" 
                          className="text-[#0284C7] dark:text-sky-400 font-semibold hover:underline hover:text-[#0369A1] dark:hover:text-sky-300 transition-colors"
                        >
                          {children}
                        </a>
                      )
                    },
                    hr: ({ node, ...props }) => (
                      <hr {...props} className="my-3.5 border-0 border-t border-gray-200 dark:border-gray-800" />
                    ),
                    table: ({ node, ...props }) => (
                      <div className="overflow-x-auto w-full pb-2 mb-4 scrollbar-thin">
                        <table {...props} className="min-w-full" />
                      </div>
                    )
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
            ) : (
              /* Live Thinking Phase with Flowchart */
              <div className="py-2">
                <div className="flex items-center gap-2.5 mb-4 text-gray-700 dark:text-purple-200">
                  <span className="w-3 h-3 rounded-full bg-[#F5C518] animate-ping"></span>
                  <span className="text-[14px] font-bold">NexAi is analyzing and executing workflow...</span>
                </div>
                {currentLogs && currentLogs.length > 0 && (
                  <div className="p-4 bg-gray-50 dark:bg-[#13131c] rounded-2xl border border-gray-200 dark:border-purple-900/40 shadow-inner">
                    <ExecutionFlowTimeline steps={currentLogs} isStreaming={true} />
                  </div>
                )}
              </div>
            )}

            {/* Stats Footer & Collapsible Execution Flow */}
            {!isUser && msg.content && (msg.total_tokens > 0 || (msg.tools_called && msg.tools_called.length > 0) || msg.response_time > 0 || (flowSteps && flowSteps.length > 0)) && (
              <div className="mt-4 pt-3 border-t border-gray-100 dark:border-purple-800/40">
                {/* Summary Pill Bar */}
                <div className="flex flex-wrap items-center justify-between gap-3 text-[12px] font-mono text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
                  <div className="flex flex-wrap items-center gap-3.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[14px]">🧠</span>
                      <span className="font-semibold text-gray-700 dark:text-gray-300">Node:</span>
                      <span>{msg.route === 'farmer' ? 'Farmer Node' : 'Omni-Agent Node'}</span>
                    </div>
                    <div className="w-px h-3 bg-gray-300 dark:bg-gray-600 hidden sm:block"></div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[14px]">🔧</span>
                      <span className="font-semibold text-gray-700 dark:text-gray-300">Tools:</span>
                      <span className="font-bold text-sky-600 dark:text-sky-400">
                        {msg.tools_called && msg.tools_called.length > 0 ? Array.from(new Set(msg.tools_called)).join(', ') : 'Direct Answer'}
                      </span>
                    </div>
                    <div className="w-px h-3 bg-gray-300 dark:bg-gray-600 hidden sm:block"></div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[14px]">⚡</span>
                      <span className="font-semibold text-gray-700 dark:text-gray-300">Tokens:</span>
                      <span className="font-bold text-amber-600 dark:text-amber-400">{msg.total_tokens?.toLocaleString() || 0}</span>
                    </div>
                    {msg.response_time > 0 && (
                      <>
                        <div className="w-px h-3 bg-gray-300 dark:bg-gray-600 hidden sm:block"></div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[14px]">⏱️</span>
                          <span className="font-semibold text-gray-700 dark:text-gray-300">Time:</span>
                          <span className="font-bold text-emerald-600 dark:text-emerald-400">{msg.response_time}s</span>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Collapsible Flow Toggle Button */}
                  {flowSteps && flowSteps.length > 0 && (
                    <button
                      onClick={() => setShowFlow(prev => !prev)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 border border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-200 text-[11px] font-bold transition-all shadow-xs cursor-pointer ml-auto"
                    >
                      <span>{showFlow ? 'Hide Flow' : 'Execution Flow'}</span>
                      <span className="px-1.5 py-0.2 rounded-full bg-yellow-100 dark:bg-purple-900/60 text-yellow-800 dark:text-purple-300 text-[10px]">
                        {flowSteps.length} steps
                      </span>
                      <ChevronDown size={13} className={`transition-transform duration-200 ${showFlow ? 'rotate-180' : ''}`} />
                    </button>
                  )}
                </div>

                {/* Collapsible Dropdown Flowchart */}
                {showFlow && flowSteps && flowSteps.length > 0 && (
                  <div className="mt-3 p-4 bg-gray-50/90 dark:bg-[#13131c] rounded-2xl border border-gray-200 dark:border-purple-900/40 animate-fade-in shadow-inner">
                    <div className="text-[11px] font-black uppercase tracking-wider text-gray-400 dark:text-purple-300 mb-3.5 flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <Activity size={13} className="text-yellow-500 dark:text-purple-400" />
                        Complete Node Execution Flowchart & Tool Trace
                      </span>
                      {msg.response_time > 0 && (
                        <span className="text-emerald-600 dark:text-emerald-400 font-mono font-bold text-[11px]">
                          Response Latency: {msg.response_time}s
                        </span>
                      )}
                    </div>
                    <ExecutionFlowTimeline steps={flowSteps} isStreaming={false} />
                  </div>
                )}
              </div>
            )}
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

        <h1 className="text-5xl sm:text-[64px] font-black tracking-tight text-gray-950 dark:text-gray-100 mb-3 leading-none">
          <span className="text-black dark:text-white">Nex</span>
          <span className="text-[#F5C518]">Ai</span>
        </h1>

        <p className="text-[15px] sm:text-[17px] text-gray-500 dark:text-gray-400 font-medium max-w-2xl mx-auto leading-relaxed mt-4">
          Your universal multi-agent assistant for web research, live cricket scores, finance, documents, and agriculture.
        </p>

        {/* Feature Tags */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
          {Object.entries(ROUTE_META).map(([key, meta]) => {
            const Icon = meta.icon
            return (
              <span key={key} className="flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-bold shadow-sm border border-gray-100 dark:border-purple-800/40 bg-white dark:bg-[#1a1a24]" style={{ color: meta.color }}>
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
              className="p-7 min-h-[140px] rounded-[24px] bg-white dark:bg-[#1a1a24] border border-gray-200 dark:border-purple-800/40 shadow-sm hover:shadow-md hover:border-yellow-400 dark:hover:border-yellow-400 transition-all group flex flex-col justify-start text-left"
            >
              <div className="flex items-center gap-4 w-full">
                <span
                  className="w-10 h-10 rounded-[14px] flex items-center justify-center shadow-sm shrink-0"
                  style={{ backgroundColor: item.bgColor, color: item.color }}
                >
                  <Icon size={18} />
                </span>
                <div className="flex-1">
                  <h3 className="font-bold text-gray-900 dark:text-gray-100 text-[15px] group-hover:text-black dark:group-hover:text-white leading-tight">
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
  const [statusLogs, setStatusLogs] = useState([])
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
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false)
  const [userGroqKey, setUserGroqKey] = useState('')
  const [activeUrl, setActiveUrl] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [uploadingPdf, setUploadingPdf] = useState(false)
  const [isListening, setIsListening] = useState(false)

  const textareaRef = useRef(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  const toggleListen = async () => {
    if (isListening) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      setIsListening(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const formData = new FormData()
        formData.append('file', audioBlob, 'audio.webm')
        
        try {
          const res = await fetch(`${API_BASE}/chat/transcribe`, {
            method: 'POST',
            body: formData,
          })
          const data = await res.json()
          if (data.text) {
             setInputText(prev => prev + (prev ? ' ' : '') + data.text.trim())
          }
        } catch (err) {
          console.error("Transcription failed:", err)
        }
        
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsListening(true)
    } catch (err) {
      console.error("Microphone access denied:", err)
      alert("Microphone access is required for voice input.")
    }
  }

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
  }, [messages, statusLogs])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 250)}px`
    }
  }, [inputText])

  const handleSelectSession = async (sessionId) => {
    setActiveSessionId(sessionId)
    setStatusLogs([])
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
    setStatusLogs([])
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
        if (data.file_id) setUploadedFiles(prev => [...prev, { id: data.file_id, name: file.name }])
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
      setStatusLogs([])
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

    const newMsgs = [...messages, { role: 'user', content: textToSend, files: uploadedFiles }]
    // NOTE: Do NOT clear uploadedFiles here — keep PDF context alive for the full session
    // setUploadedFiles([])  ← removed: this was causing PDF to be forgotten on 2nd question
    setMessages(newMsgs)
    setStreaming(true)
    setStatusLogs([{ node: 'system', msg: 'Agent is thinking...' }])

    const controller = new AbortController()
    setAbortController(controller)

    const assistantIndex = newMsgs.length
    setMessages([...newMsgs, { role: 'assistant', content: '', route: 'general', sources: [] }])

    const params = new URLSearchParams({
      session_id: activeSessionId,
      message: textToSend,
      file_ids: uploadedFiles.map(f => f.id).join(','),
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
                if (data.message) {
                  setStatusLogs(prev => {
                    if (prev.length > 0 && prev[prev.length - 1].msg === data.message) return prev;
                    return [...prev, { node: data.node || 'system', msg: data.message }];
                  });
                }
              } else if (currentEvent === 'result' || data.answer) {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[assistantIndex] = {
                    role: 'assistant',
                    content: data.answer,
                    route: data.route || 'general',
                    sources: data.sources || [],
                    total_tokens: data.total_tokens || 0,
                    tools_called: data.tools_called || [],
                    response_time: data.response_time || 0,
                    execution_flow: data.execution_flow || statusLogs,
                  }
                  return updated
                })
                setStatusLogs([])
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
                setStatusLogs([])
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
      setStatusLogs([])
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
                  className="w-full px-4 py-3.5 rounded-xl bg-white dark:bg-[#1a1a24] border border-gray-200 dark:border-purple-800/40 text-[14px] font-bold text-gray-900 dark:text-gray-100 shadow-sm outline-none focus:border-[#F5C518] focus:ring-2 focus:ring-yellow-400/20 transition-all leading-normal h-12 pr-10"
                />
                <Eye size={16} className="absolute right-4 top-[14px] text-gray-400" />
              </div>
              <p className="text-[11px] text-gray-400 mt-2 font-medium">Leave blank to use built-in key</p>
              <div className="mt-1.5">
                {userGroqKey.trim() ? (
                  <span className="text-[11px] font-bold text-emerald-700 dark:text-emerald-500 flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Custom key active</span>
                ) : (
                  <span className="text-[11px] font-bold text-amber-700 dark:text-amber-500 flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[#F5C518]"></span> Built-in key active</span>
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
                className="w-full px-4 py-3.5 rounded-xl bg-white dark:bg-[#1a1a24] border border-gray-200 dark:border-purple-800/40 text-[14px] font-medium text-gray-900 dark:text-gray-100 shadow-sm outline-none focus:border-[#F5C518] focus:ring-2 focus:ring-yellow-400/20 transition-all leading-normal h-12"
              />
            </div>
          </div>

          <div className="pt-6 border-t border-gray-200 dark:border-purple-800/40">
            <div className="flex items-center justify-between mb-4">
              <span className="text-[12px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <Clock size={14} className="text-[#F5C518]" /> Chat History ({sessions.length})
              </span>
              <button onClick={loadSessions} className="p-1.5 hover:bg-gray-100 dark:hover:bg-purple-900/30 rounded-lg transition-colors"><RefreshCw size={14} className="text-gray-400 hover:text-black dark:hover:text-white" /></button>
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
                    currentLogs={statusLogs}
                    onEdit={handleEditMessage} 
                  />
                ))}
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
                    {uploadedFiles.length > 0 && <span className="w-5 h-5 flex items-center justify-center rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-400 text-[11px] ml-1">{uploadedFiles.length}</span>}
                  </button>
                </div>
                {/* PDF active badge — shows filename and allows clearing */}
                {uploadedFiles.length > 0 && (
                  <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300 text-[12px] font-medium max-w-[200px]">
                    <span className="truncate" title={uploadedFiles.map(f => f.name).join(', ')}>
                      📄 {uploadedFiles[uploadedFiles.length - 1].name}
                    </span>
                    <button
                      onClick={() => setUploadedFiles([])}
                      className="ml-1 text-emerald-500 hover:text-red-500 dark:hover:text-red-400 transition-colors font-bold flex-shrink-0"
                      title="Remove PDF"
                    >×</button>
                  </div>
                )}
                
                <button onClick={() => setFarmerMode(!farmerMode)} className={`flex items-center gap-2 px-4 py-2 rounded-full text-[14px] font-bold transition-colors ${farmerMode ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 shadow-sm' : 'hover:bg-gray-100 dark:hover:bg-purple-900/30 text-gray-600 dark:text-gray-300'}`}>
                  <Sprout size={16} className={farmerMode ? 'text-emerald-500 dark:text-emerald-400' : ''} /> Farmer Mode 
                </button>
                
                <div className="relative flex items-center">
                  <button 
                    onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                    className="flex items-center gap-2 px-4 py-2 rounded-full text-[14px] font-bold text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-purple-900/30 transition-colors relative"
                  >
                    <Cpu size={16} className="text-[#F5C518] dark:text-purple-400" />
                    {agentModel.split('/')[1] || 'Model'} <ChevronDown size={16} className={`text-gray-400 dark:text-gray-500 transition-transform ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {isModelDropdownOpen && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={() => setIsModelDropdownOpen(false)}></div>
                      <div className="absolute bottom-full left-0 mb-2 w-56 bg-white dark:bg-[#1a1a24] border border-gray-100 dark:border-purple-800/40 rounded-2xl shadow-xl z-50 overflow-hidden transform origin-bottom-left transition-all">
                        <div className="p-2 border-b border-gray-50 dark:border-purple-900/30">
                          <p className="text-[11px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500 px-3 py-1">Select Brain</p>
                        </div>
                        <div className="p-1">
                          {[
                            'openai/gpt-oss-120b',
                            'qwen/qwen3.6-27b',
                            'openai/gpt-oss-20b',
                            'qwen/qwen3.8-27b'
                          ].map(model => (
                            <button
                              key={model}
                              onClick={() => {
                                setAgentModel(model);
                                setIsModelDropdownOpen(false);
                              }}
                              className={`w-full text-left px-3 py-2.5 rounded-xl text-[13px] font-medium flex items-center justify-between transition-colors ${
                                agentModel === model 
                                  ? 'bg-purple-50 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' 
                                  : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-[#252532]'
                              }`}
                            >
                              <span>{model.split('/')[1]}</span>
                              {agentModel === model && <Check size={14} className="text-purple-600 dark:text-purple-400" />}
                            </button>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <button 
                  onClick={toggleListen}
                  className={`w-11 h-11 rounded-full border flex items-center justify-center transition-colors shadow-sm ${
                    isListening 
                      ? 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800/40 text-red-500 animate-pulse' 
                      : 'bg-white dark:bg-[#1a1a24] border-gray-200 dark:border-purple-800/40 hover:bg-gray-50 dark:hover:bg-purple-900/30 text-gray-600 dark:text-gray-400'
                  }`}
                  title={isListening ? "Stop listening" : "Start speaking"}
                >
                  {isListening ? <Square size={16} fill="currentColor" /> : <Mic size={18} />}
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
