'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import {
  Upload,
  FileText,
  Loader2,
  Sparkles,
  Download,
  CheckCircle2,
  Eye,
  EyeOff,
  Brain,
  Code2,
  Zap,
  X,
  Maximize2,
  ChevronDown,
  Star,
  ArrowUpRight,
} from 'lucide-react'
import { twMerge } from 'tailwind-merge'
import clsx from 'clsx'
import { RainbowButton } from '@/components/ui/rainbow-button'

// Helper for class merging
function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs))
}

// Tag colors by category
const TAG_STYLES = {
  violet:  'bg-violet-500/15 text-violet-300 border border-violet-500/25',
  pink:    'bg-pink-500/15 text-pink-300 border border-pink-500/25',
  sky:     'bg-[#8ad4ff]/15 text-[#8ad4ff] border border-[#8ad4ff]/30',
  emerald: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25',
  amber:   'bg-amber-500/15 text-amber-300 border border-amber-500/25',
  teal:    'bg-teal-500/15 text-teal-300 border border-teal-500/25',
  rose:    'bg-rose-500/15 text-rose-300 border border-rose-500/25',
} as const

function TagRow({ label, tags, color }: { label: string; tags: string[]; color: keyof typeof TAG_STYLES }) {
  const filtered = tags.filter(Boolean)
  if (!filtered.length) return null
  return (
    <div className="flex items-start gap-2">
      <span className="text-[10px] text-white/25 uppercase tracking-widest w-14 shrink-0 pt-0.5 font-medium">{label}</span>
      <div className="flex flex-wrap gap-1">
        {filtered.map((tag, i) => (
          <span key={i} className={`text-[11px] px-2 py-0.5 rounded-full ${TAG_STYLES[color]}`}>{tag}</span>
        ))}
      </div>
    </div>
  )
}

// ─── Trending Papers ──────────────────────────────────────────────────────────

interface TrendingPaper {
  id: string
  title: string
  abstract: string
  authors: string
  publishedAt: string
  upvotes: number
  githubUrl: string
  githubStars: number
  keywords: string[]
  arxivUrl: string
  hfUrl: string
  thumbnail: string
}

function getTaskStyle(task: string): string {
  const t = task.toLowerCase()
  if (t.includes('vision') || t.includes('image') || t.includes('object') || t.includes('segmentation') || t.includes('detection'))
    return 'bg-indigo-500/15 text-indigo-300 border-indigo-500/25'
  if (t.includes('language') || t.includes('nlp') || t.includes('text') || t.includes('translation') || t.includes('question') || t.includes('summar'))
    return 'bg-violet-500/15 text-violet-300 border-violet-500/25'
  if (t.includes('reinforcement') || t.includes('robot') || t.includes('control') || t.includes('planning'))
    return 'bg-amber-500/15 text-amber-300 border-amber-500/25'
  if (t.includes('generat') || t.includes('diffusion') || t.includes('gan') || t.includes('synthesis'))
    return 'bg-pink-500/15 text-pink-300 border-pink-500/25'
  if (t.includes('graph') || t.includes('node') || t.includes('link'))
    return 'bg-teal-500/15 text-teal-300 border-teal-500/25'
  if (t.includes('audio') || t.includes('speech') || t.includes('voice') || t.includes('sound'))
    return 'bg-orange-500/15 text-orange-300 border-orange-500/25'
  if (t.includes('classif') || t.includes('recogni'))
    return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25'
  if (t.includes('asr') || t.includes('tts') || t.includes('llm') || t.includes('vlm') || t.includes('mllm'))
    return 'bg-orange-500/15 text-orange-300 border-orange-500/25'
  if (t.includes('efficient') || t.includes('compress') || t.includes('quant') || t.includes('prun') || t.includes('distill'))
    return 'bg-lime-500/15 text-lime-300 border-lime-500/25'
  if (t.includes('multimodal') || t.includes('multi-modal') || t.includes('cross-modal'))
    return 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/25'
  // Default: rotate through a set of vibrant colors based on text hash
  const colors = [
    'bg-violet-500/15 text-violet-300 border-violet-500/25',
    'bg-pink-500/15 text-pink-300 border-pink-500/25',
    'bg-amber-500/15 text-amber-300 border-amber-500/25',
    'bg-emerald-500/15 text-emerald-300 border-emerald-500/25',
    'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/25',
    'bg-lime-500/15 text-lime-300 border-lime-500/25',
  ]
  const hash = task.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0)
  return colors[hash % colors.length]
}

function PaperCardSkeleton() {
  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl overflow-hidden space-y-3 animate-pulse">
      <div className="w-full aspect-[2/1] bg-white/10 rounded-t-2xl" />
      <div className="p-5 space-y-3">
      <div className="flex gap-2 items-center">
        <div className="h-5 w-10 bg-white/10 rounded" />
        <div className="h-5 w-16 bg-white/10 rounded-full" />
        <div className="h-5 w-14 bg-white/10 rounded-full" />
        <div className="h-4 w-10 bg-white/5 rounded ml-auto" />
      </div>
      <div className="h-4 bg-white/10 rounded w-5/6" />
      <div className="h-4 bg-white/10 rounded w-3/5" />
      <div className="h-3 bg-white/5 rounded w-2/5" />
      <div className="space-y-1.5 pt-1">
        <div className="h-3 bg-white/5 rounded w-full" />
        <div className="h-3 bg-white/5 rounded w-5/6" />
        <div className="h-3 bg-white/5 rounded w-4/6" />
      </div>
      <div className="flex gap-2 pt-2 border-t border-white/5">
        <div className="h-7 w-20 bg-white/5 rounded-lg" />
        <div className="h-7 w-36 bg-white/5 rounded-lg ml-auto" />
      </div>
      </div>{/* end p-5 body */}
    </div>
  )
}

function PaperCard({ paper, onConvert }: { paper: TrendingPaper; onConvert: (url: string) => void }) {
  return (
    <motion.div
      whileHover={{ y: -5, borderColor: 'rgba(255,255,255,0.22)' }}
      transition={{ type: 'spring', stiffness: 320, damping: 22 }}
      className="group relative bg-white/[0.04] border border-white/10 rounded-2xl flex flex-col overflow-hidden"
    >
      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{ boxShadow: '0 8px 40px -8px rgba(138,212,255,0.18)' }} />

      {/* Thumbnail */}
      {paper.thumbnail && (
        <div className="relative w-full aspect-[2/1] overflow-hidden rounded-t-2xl bg-white/5 shrink-0">
          <img
            src={paper.thumbnail}
            alt=""
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            loading="lazy"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#050505]/80 via-transparent to-transparent" />
        </div>
      )}

      {/* Card body */}
      <div className="flex flex-col gap-4 p-7 flex-1">

      {/* Top row: date + keywords + github stars */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {paper.publishedAt && (
          <span className="text-xs font-mono bg-[#ffd78a]/10 text-[#ffd78a]/60 px-2.5 py-1 rounded shrink-0">
            {paper.publishedAt}
          </span>
        )}
        {paper.keywords.slice(0, 2).map(kw => (
          <span key={kw} className={`text-xs px-2.5 py-1 rounded-full border ${getTaskStyle(kw)}`}>
            {kw.length > 22 ? kw.slice(0, 21) + '…' : kw}
          </span>
        ))}
        {paper.githubUrl ? (
          <a
            href={paper.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            className="ml-auto flex items-center gap-1 text-xs text-white/50 hover:text-white/80 transition-colors shrink-0"
            title="GitHub repository"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
            </svg>
            {paper.githubStars > 0 ? (
              <span>{paper.githubStars >= 1000 ? (paper.githubStars / 1000).toFixed(1) + 'k' : paper.githubStars}</span>
            ) : null}
          </a>
        ) : paper.upvotes > 0 ? (
          <div className="ml-auto flex items-center gap-1 text-xs text-[#ffd78a]/70 shrink-0" title="Upvotes on HuggingFace">
            <Star className="w-3.5 h-3.5 fill-[#ffd78a]/40" />
            {paper.upvotes}
          </div>
        ) : null}
      </div>

      {/* Title */}
      <h3 className="text-base font-bold text-white leading-snug group-hover:text-[#8ad4ff] transition-colors line-clamp-2">
        {paper.title}
      </h3>

      {/* Authors */}
      <p className="text-xs text-white/35 font-medium -mt-1">{paper.authors}</p>

      {/* Abstract */}
      <p className="text-sm text-white/55 leading-relaxed line-clamp-3 flex-1">{paper.abstract}</p>

      {/* Bottom actions */}
      <div className="flex items-center gap-2 pt-3 border-t border-white/5">
        <a
          href={paper.hfUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={e => e.stopPropagation()}
          className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/70 transition-colors px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10"
        >
          <ArrowUpRight className="w-3.5 h-3.5" />
          HF Paper
        </a>
        <button
          onClick={() => onConvert(paper.arxivUrl)}
          className="ml-auto flex items-center gap-2 text-sm font-semibold bg-[#8ad4ff]/15 hover:bg-[#8ad4ff]/25 text-[#8ad4ff] border border-[#8ad4ff]/30 px-4 py-2 rounded-lg transition-colors"
        >
          Convert to Notebook
          <img src="/paper2codelogo.png" className="w-4 h-4 object-contain" alt="" />
        </button>
      </div>
      </div>{/* end card body */}
    </motion.div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

// Hero Component
const Hero = () => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.8 }}
    className="text-center mb-8 pt-6 relative z-10"
  >
    <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-[#8ad4ff] via-white to-purple-200 mb-2 drop-shadow-2xl">
      Paper to Notebook
    </h1>
    <p className="text-sm text-neutral-400 max-w-2xl mx-auto">
      Upload a research paper PDF - get a runnable PyTorch notebook
    </p>
  </motion.div>
)

function TrendingButton({ onClick }: { onClick: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1 }}
      className="relative group/trendbtn z-[60]"
    >
      <RainbowButton onClick={onClick} className="flex items-center gap-2 text-sm h-9 px-5 rounded-full">
        <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
          <polyline points="17 6 23 6 23 12" />
        </svg>
        Trending Papers
        <ChevronDown className="w-3.5 h-3.5" />
      </RainbowButton>
    </motion.div>
  )
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [arxivUrl, setArxivUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [selectedModel, setSelectedModel] = useState('google/gemini-2.5-pro')
  const [availableModels, setAvailableModels] = useState<Array<{id: string, name: string, provider: string, context_length?: number, max_output?: number, price_per_1m_tokens?: number}>>([])
  const [modelSearch, setModelSearch] = useState('')
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const modelDropdownRef = useRef<HTMLDivElement>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const [bannerVisible, setBannerVisible] = useState(true)

  // Progress tracking
  const [steps, setSteps] = useState<Array<{ name: string; detail: string }>>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<boolean[]>([false, false, false, false])

  // Results
  const [draftJobId, setDraftJobId] = useState('')
  const [finalJobId, setFinalJobId] = useState('')
  const [thinking, setThinking] = useState('')
  const [activities, setActivities] = useState<any[]>([])
  const [showDone, setShowDone] = useState(false)
  const [thinkingExpanded, setThinkingExpanded] = useState(false)

  // Trending papers
  const [trendingPapers, setTrendingPapers] = useState<TrendingPaper[]>([])
  const [trendingLoading, setTrendingLoading] = useState(true)
  const [trendingPeriod, setTrendingPeriod] = useState<'day' | 'week' | 'month'>('day')
  const [trendingSort, setTrendingSort] = useState<'stars' | 'new'>('stars')
  const trendingSectionRef = useRef<HTMLElement>(null)
  const arxivInputRef = useRef<HTMLInputElement>(null)

  // Paper info for engaging wait experience

  // arXiv instant metadata
  const [arxivMeta, setArxivMeta] = useState<{ title: string; authors: string; abstract: string; categories: string[]; published: string; venue: string } | null>(null)
  const [showArxivExpanded, setShowArxivExpanded] = useState(false)
  const [panelExpanded, setPanelExpanded] = useState(false)

  const CATEGORY_LABELS: Record<string, string> = {
    'cs.LG': 'Machine Learning', 'cs.CV': 'Computer Vision', 'cs.CL': 'NLP',
    'cs.AI': 'AI', 'cs.NE': 'Neural Networks', 'cs.RO': 'Robotics',
    'stat.ML': 'Statistics ML', 'cs.IR': 'Information Retrieval',
    'cs.CR': 'Cryptography', 'cs.DC': 'Distributed Computing',
    'math.OC': 'Optimization', 'eess.IV': 'Image Processing',
  }

  const fetchArxivMeta = async (url: string) => {
    const match = url.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+(?:v\d+)?)/)
    if (!match) return
    try {
      const res = await fetch(`/api/arxiv?id=${match[1]}`)
      const text = await res.text()
      const xml = new DOMParser().parseFromString(text, 'text/xml')
      const entry = xml.querySelector('entry')
      if (!entry) return
      const title = entry.querySelector('title')?.textContent?.replace(/\s+/g, ' ').trim() || ''
      const authors = Array.from(entry.querySelectorAll('author name')).map(a => a.textContent).join(', ')
      const abstract = entry.querySelector('summary')?.textContent?.replace(/\s+/g, ' ').trim() || ''
      const categories = Array.from(entry.querySelectorAll('category')).map(c => c.getAttribute('term') || '').filter(Boolean)
      const published = entry.querySelector('published')?.textContent?.slice(0, 10) || ''
      const journalRef = entry.querySelector('journal_ref')?.textContent?.trim() || ''
      const comment = entry.querySelector('comment')?.textContent?.trim() || ''
      const venue = journalRef || (comment.match(/(?:ICLR|NeurIPS|ICML|CVPR|ICCV|ECCV|ACL|EMNLP|AAAI|IJCAI|NAACL|INTERSPEECH|ICASSP)[^,.]*/i)?.[0] || '')
      setArxivMeta({ title, authors, abstract, categories, published, venue })
    } catch {}
  }

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const thinkingScrollRef = useRef<HTMLDivElement>(null)
  const userScrolledUpRef = useRef(false)

  // Typewriter effect
  const [displayedThinking, setDisplayedThinking] = useState('')
  const pendingCharsRef = useRef('')
  const typeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const newChars = thinking.slice(displayedThinking.length + pendingCharsRef.current.length)
    if (newChars) pendingCharsRef.current += newChars

    if (!typeTimerRef.current) {
      typeTimerRef.current = setInterval(() => {
        if (pendingCharsRef.current.length === 0) {
          clearInterval(typeTimerRef.current!)
          typeTimerRef.current = null
          return
        }
        const batch = pendingCharsRef.current.slice(0, 4)
        pendingCharsRef.current = pendingCharsRef.current.slice(4)
        setDisplayedThinking(prev => {
          const next = prev + batch
          // Auto-scroll only if user hasn't scrolled up
          if (!userScrolledUpRef.current && thinkingScrollRef.current) {
            thinkingScrollRef.current.scrollTop = thinkingScrollRef.current.scrollHeight
          }
          return next
        })
      }, 18)
    }
  }, [thinking])

  const handleThinkingScroll = useCallback(() => {
    if (!thinkingScrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = thinkingScrollRef.current
    userScrolledUpRef.current = scrollHeight - scrollTop - clientHeight > 60
  }, [])

  // Fetch trending papers when period changes
  useEffect(() => {
    setTrendingLoading(true)
    setTrendingPapers([])
    fetch(`/api/trending-papers?period=${trendingPeriod}`)
      .then(r => r.json())
      .then(data => { setTrendingPapers(data.papers || []); setTrendingLoading(false) })
      .catch(() => setTrendingLoading(false))
  }, [trendingPeriod])

  const handleConvertPaper = useCallback((url: string) => {
    setArxivUrl(url)
    setSelectedFile(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
    setTimeout(() => arxivInputRef.current?.focus(), 600)
  }, [])

  // Load API key from localStorage and check for pending file/URL
  useEffect(() => {
    const savedKey = localStorage.getItem('openrouter_api_key')
    if (savedKey) setApiKey(savedKey)

    // Fetch available models from backend
    fetch(`${API_URL}/api/models`)
      .then(r => r.json())
      .then(data => {
        setAvailableModels(data.models || [])
        if (data.default) setSelectedModel(data.default)
      })
      .catch(() => {})

    // Check for pending file from landing page
    const pendingFileData = sessionStorage.getItem('pendingFile')
    const pendingFileName = sessionStorage.getItem('pendingFileName')
    if (pendingFileData && pendingFileName) {
      // Convert base64 back to File
      fetch(pendingFileData)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], pendingFileName, { type: 'application/pdf' })
          setSelectedFile(file)
        })
      // Clear from sessionStorage
      sessionStorage.removeItem('pendingFile')
      sessionStorage.removeItem('pendingFileName')
    }

    // Check for pending arXiv URL from landing page
    const pendingUrl = sessionStorage.getItem('pendingArxivUrl')
    if (pendingUrl) {
      setArxivUrl(pendingUrl)
      sessionStorage.removeItem('pendingArxivUrl')
    }
  }, [])

  // Close model dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // File drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      setSelectedFile(file)
      setError('')
      setDraftJobId('')
      setFinalJobId('')
      setShowDone(false)
    } else {
      setError('Please upload a PDF file')
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setArxivUrl('') // Clear arXiv URL if file is selected
      setError('')
      setDraftJobId('')
      setFinalJobId('')
      setShowDone(false)
    }
  }

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const key = e.target.value
    setApiKey(key)
    localStorage.setItem('openrouter_api_key', key)
  }

  const handleGenerate = async () => {
    // Check if either file or arXiv URL is provided
    if ((!selectedFile && !arxivUrl.trim()) || !apiKey.trim()) {
      setError('Please select a PDF or paste an arXiv link, and enter your API key')
      return
    }

    setIsGenerating(true)
    setError('')
    setDraftJobId('')
    setFinalJobId('')
    setShowDone(false)
    setActivities([])
    setThinking('')
    setDisplayedThinking('')
    setArxivMeta(null)
    pendingCharsRef.current = ''
    userScrolledUpRef.current = false
    if (typeTimerRef.current) { clearInterval(typeTimerRef.current); typeTimerRef.current = null }
    setCurrentStep(0)

    if (arxivUrl.trim()) fetchArxivMeta(arxivUrl.trim())
    setSteps([
      { name: 'Analyze paper', detail: '' },
      { name: 'Design implementation', detail: '' },
      { name: 'Generate notebook', detail: '' },
      { name: 'Validate code', detail: '' },
    ])
    setCompletedSteps([false, false, false, false])

    const formData = new FormData()
    formData.append('api_key', apiKey.trim())
    formData.append('model', selectedModel)

    // Determine which endpoint to use
    let endpoint = `${API_URL}/api/generate`
    if (arxivUrl.trim()) {
      endpoint = `${API_URL}/api/generate-from-arxiv`
      formData.append('arxiv_url', arxivUrl.trim())
    } else if (selectedFile) {
      formData.append('file', selectedFile)
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        if (response.status === 429) throw new Error('Rate limit reached. Please try again later.')
        if (response.status === 413) throw new Error('PDF too large. Maximum is 30 MB.')
        throw new Error(`Server error (${response.status})`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('Failed to read response')
      }

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          if (part.startsWith('event: thinking')) {
            const dataLine = part.split('\n').find(l => l.startsWith('data: '))
            if (dataLine) {
              const data = JSON.parse(dataLine.slice(6))
              setThinking(prev => (prev + data.text).slice(-2000))
            }
          } else if (part.startsWith('event: progress')) {
            const dataLine = part.split('\n').find(l => l.startsWith('data: '))
            if (dataLine) {
              const data = JSON.parse(dataLine.slice(6))
              handleProgress(data)
            }
          } else if (part.startsWith('event: draft_ready')) {
            const dataLine = part.split('\n').find(l => l.startsWith('data: '))
            if (dataLine) {
              const data = JSON.parse(dataLine.slice(6))
              setDraftJobId(data.job_id)
            }
          } else if (part.startsWith('event: complete')) {
            const dataLine = part.split('\n').find(l => l.startsWith('data: '))
            if (dataLine) {
              const data = JSON.parse(dataLine.slice(6))
              handleComplete(data.job_id)
            }
          } else if (part.startsWith('event: error')) {
            const dataLine = part.split('\n').find(l => l.startsWith('data: '))
            if (dataLine) {
              const data = JSON.parse(dataLine.slice(6))
              throw new Error(data.error)
            }
          }
        }
      }
    } catch (err: any) {
      setError(err.message || 'Connection lost. Please try again.')
      setIsGenerating(false)
    }
  }

  const handleProgress = (data: any) => {
    const { step, detail, extra } = data

    // Reset thinking stream when a new stage begins
    setCurrentStep(prev => {
      if (step > prev) {
        setThinking('')
        setDisplayedThinking('')
        pendingCharsRef.current = ''
        userScrolledUpRef.current = false
        setThinkingExpanded(false)
      }
      return step
    })

    // Update step details
    setSteps(prev => {
      const newSteps = [...prev]
      newSteps[step - 1] = { ...newSteps[step - 1], detail }
      return newSteps
    })

    // Mark previous steps as completed
    setCompletedSteps(prev => {
      const newCompleted = [...prev]
      for (let i = 0; i < step - 1; i++) {
        newCompleted[i] = true
      }
      return newCompleted
    })

    // Add activity cards
    if (extra) {
      if (extra.type === 'analysis') {
        setActivities(prev => [...prev, {
          type: 'analysis',
          label: 'Paper Analysis',
          title: extra.title,
          content: extra,
        }])
      } else if (extra.type === 'design') {
        setActivities(prev => [...prev, {
          type: 'design',
          label: 'Architecture',
          title: extra.notebook_title || 'Implementation Plan',
          content: extra,
        }])
      } else if (extra.type === 'cells_generated') {
        setActivities(prev => [...prev, {
          type: 'cells',
          label: 'Notebook Generated',
          title: `${extra.num_cells} cells (${extra.code_cells} code)`,
          content: extra,
        }])
      }
    }
  }

  // Lock body scroll when panel is expanded
  useEffect(() => {
    document.body.style.overflow = panelExpanded ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [panelExpanded])

  const handleComplete = (jobId: string) => {
    setFinalJobId(jobId)
    setCompletedSteps([true, true, true, true])
    setIsGenerating(false)
    setShowDone(true)
    setThinking('')
  }

  const stepIcons = [Brain, Code2, Sparkles, CheckCircle2]

  return (
    <div className="min-h-screen bg-[#050505] text-white overflow-x-hidden relative font-sans">
      {/* Banner */}
      {bannerVisible && (
        <div className="fixed top-0 w-full z-50 bg-[#8ad4ff] py-1.5">
          <div className="max-w-7xl mx-auto px-6 flex items-center justify-center gap-4 relative">
            <span className="text-sm font-medium text-black">Try our new Personalized AI tutor</span>
            <a
              href="https://vizz.vizuara.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-full text-sm font-medium text-black transition-colors"
            >
              Vizz-AI →
            </a>
            <button
              onClick={() => setBannerVisible(false)}
              className="absolute right-6 text-black hover:text-black/80 transition-colors"
              aria-label="Close banner"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Background Gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[120px] animate-pulse-glow" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[120px] animate-pulse-glow" />
        <div className="absolute top-[40%] left-[50%] -translate-x-1/2 w-[60%] h-[60%] bg-purple-900/5 rounded-full blur-[150px]" />
      </div>

      <div className="container mx-auto px-4 pt-16 pb-6 relative z-10 max-w-6xl">
        <Hero />
        <div className="flex justify-center mb-6 -mt-2 relative z-[60]">
          <TrendingButton onClick={() => trendingSectionRef.current?.scrollIntoView({ behavior: 'smooth' })} />
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Upload & Controls */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-4 flex flex-col min-h-[600px]"
          >
            {/* Upload Zone */}
            <div className="rounded-2xl transition-all duration-300 flex-1 flex items-center">
              <div
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
                className={cn(
                  "border-2 border-dashed rounded-2xl p-12 transition-all duration-300 cursor-pointer w-full flex items-center justify-center",
                  "bg-white/5 backdrop-blur-sm border-white/10 hover:border-white/30"
                )}
              >
                <input
                  id="file-input"
                  type="file"
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                <AnimatePresence mode="wait">
                  {!selectedFile ? (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="flex flex-col items-center text-center space-y-4"
                    >
                      <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center border border-white/10 animate-float">
                        <Upload className="w-8 h-8 text-[#8ad4ff]" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white/90 mb-1">Drop your PDF here</h3>
                        <p className="text-sm text-white/50">or click to browse · Max 30 MB</p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="file"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="flex flex-col items-center text-center space-y-3 w-full px-8"
                    >
                      <div className="w-16 h-16 rounded-full bg-[#8ad4ff]/20 flex items-center justify-center border border-[#8ad4ff]/30 shadow-[0_0_30px_-5px_rgba(99,102,241,0.3)]">
                        <FileText className="w-8 h-8 text-[#8ad4ff]" />
                      </div>
                      <div className="w-full overflow-hidden">
                        <h3 className="text-base font-medium text-white truncate">{selectedFile.name}</h3>
                        <p className="text-sm text-[#8ad4ff] mt-1">Ready to analyze</p>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); setSelectedFile(null) }}
                        className="text-sm text-white/40 hover:text-white underline decoration-white/20 hover:decoration-white transition-colors"
                      >
                        Remove file
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* OR Divider */}
            <div className="flex items-center gap-3 text-white/30 text-sm">
              <div className="flex-1 h-px bg-white/10"></div>
              <span>OR</span>
              <div className="flex-1 h-px bg-white/10"></div>
            </div>

            {/* arXiv URL Input */}
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4">
              <label className="text-sm font-medium text-white/60 mb-2 block">Paste arXiv Link</label>
              <input
                type="text"
                value={arxivUrl}
                onChange={(e) => {
                  setArxivUrl(e.target.value)
                  if (e.target.value.trim()) {
                    setSelectedFile(null) // Clear file if URL is entered
                  }
                }}
                ref={arxivInputRef}
                placeholder="https://arxiv.org/pdf/..."
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm focus:border-white/30 focus:outline-none transition-colors"
              />
            </div>

            {/* API Key Input */}
            <div className="bg-white/5 backdrop-blur-md border-2 border-[#8ad4ff]/40 rounded-xl p-3 space-y-3">
              <label className="text-sm font-medium text-white/80">OpenRouter API Key</label>
              <div className="flex gap-2">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={handleApiKeyChange}
                  placeholder="sk-or-..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm font-mono focus:border-white/30 focus:outline-none transition-colors"
                />
                <button
                  onClick={() => setShowKey(!showKey)}
                  className="bg-white/5 border border-white/10 rounded-lg px-4 hover:bg-white/10 transition-colors"
                >
                  {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-white/50">
                <a href="https://openrouter.ai/keys" target="_blank" rel="noopener" className="text-[#8ad4ff] hover:text-[#8ad4ff]">
                  Get an API key
                </a> from OpenRouter - supports 200+ models
              </p>
            </div>

            {/* Model Selector */}
            <div ref={modelDropdownRef} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-3 space-y-2 relative">
              <label className="text-sm font-medium text-white/80">Model</label>
              <button
                type="button"
                onClick={() => { setModelDropdownOpen(!modelDropdownOpen); setModelSearch('') }}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm text-left text-white focus:border-white/30 focus:outline-none transition-colors flex items-center justify-between"
              >
                <span className="truncate">
                  {availableModels.find(m => m.id === selectedModel)?.name || selectedModel}
                </span>
                <svg className={`w-4 h-4 ml-2 transition-transform ${modelDropdownOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              </button>
              {modelDropdownOpen && (
                <div className="absolute left-0 right-0 bottom-full mb-1 z-50 bg-[#0a0a0a] border border-white/20 rounded-xl shadow-2xl overflow-hidden flex flex-col" style={{maxHeight: '300px'}}>
                  <div className="flex-1 overflow-y-auto">
                    {(() => {
                      const query = modelSearch.toLowerCase()
                      const filtered = availableModels.filter(m =>
                        m.name.toLowerCase().includes(query) || m.id.toLowerCase().includes(query) || m.provider.toLowerCase().includes(query)
                      )
                      const grouped = filtered.reduce((acc, m) => {
                        const p = m.provider
                        if (!acc[p]) acc[p] = []
                        acc[p].push(m)
                        return acc
                      }, {} as Record<string, typeof availableModels>)
                      const sortedProviders = Object.keys(grouped).sort()

                      if (sortedProviders.length === 0) {
                        return <div className="px-3 py-4 text-sm text-white/40 text-center">No models found</div>
                      }

                      return sortedProviders.map(provider => (
                        <div key={provider}>
                          <div className="px-3 py-1 text-[10px] font-semibold text-white/40 uppercase tracking-wider bg-white/5 sticky top-0">
                            {provider}
                          </div>
                          {grouped[provider].map(m => (
                            <button
                              key={m.id}
                              type="button"
                              onClick={() => { setSelectedModel(m.id); setModelDropdownOpen(false) }}
                              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-white/10 transition-colors flex items-center justify-between gap-2 ${m.id === selectedModel ? 'bg-white/10 text-[#8ad4ff]' : 'text-white/80'}`}
                            >
                              <span className="truncate">{m.name}</span>
                              <span className="text-[10px] text-white/30 whitespace-nowrap flex-shrink-0">
                                {m.context_length ? `${Math.round(m.context_length / 1000)}k` : ''}
                                {m.price_per_1m_tokens !== undefined ? ` · $${m.price_per_1m_tokens}` : ''}
                              </span>
                            </button>
                          ))}
                        </div>
                      ))
                    })()}
                  </div>
                  <div className="p-2 border-t border-white/10 flex-shrink-0">
                    <input
                      type="text"
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      placeholder="Search models..."
                      autoFocus
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white placeholder-white/40 focus:border-white/30 focus:outline-none"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={(!selectedFile && !arxivUrl.trim()) || !apiKey.trim() || isGenerating}
              className={cn(
                "w-full rounded-xl transition-all duration-300 relative overflow-hidden group p-[2px] bg-gradient-to-r from-[#ffd78a] via-[#8ad4ff] to-[#ffa8ff] flex-shrink-0",
                isGenerating
                  ? "shadow-xl shadow-[#8ad4ff]/40 animate-pulse cursor-not-allowed"
                  : (!selectedFile && !arxivUrl.trim()) || !apiKey.trim()
                  ? "opacity-50 cursor-not-allowed"
                  : "hover:shadow-xl hover:shadow-[#8ad4ff]/40 active:scale-[0.98]"
              )}
            >
              {isGenerating ? (
                <div className="w-full h-full rounded-xl bg-[#0a0a0a] flex items-center justify-center gap-3 font-semibold text-base text-white py-2.5">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating Notebook...
                </div>
              ) : (!selectedFile && !arxivUrl.trim()) || !apiKey.trim() ? (
                <div className="w-full h-full rounded-xl bg-[#0a0a0a] flex items-center justify-center gap-3 font-semibold text-base text-white/40 py-2.5">
                  <Zap className="w-5 h-5" />
                  Generate Notebook
                </div>
              ) : (
                <div className="w-full h-full rounded-xl bg-[#0a0a0a] flex items-center justify-center gap-3 font-semibold text-base text-white py-2.5">
                  <Zap className="w-5 h-5" />
                  Generate Notebook
                </div>
              )}
            </button>
          </motion.div>

          {/* Right Column: Progress & Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className={panelExpanded ? "fixed top-4 bottom-4 left-[24%] w-[52%] z-[90]" : "relative"}
          >
            {panelExpanded && (
              <div className="fixed inset-0 z-[80] bg-black/60" onClick={() => setPanelExpanded(false)} />
            )}
            <div className={`rounded-2xl p-[2px] bg-gradient-to-r from-[#ffd78a] via-[#8ad4ff] to-[#ffa8ff] shadow-2xl relative z-[90] ${panelExpanded ? "h-full flex flex-col" : ""}`}>
              <div className={`rounded-[14px] bg-[#0a0a0a] backdrop-blur-xl overflow-hidden relative ${panelExpanded ? "flex-1 flex flex-col" : "h-full min-h-[650px]"}`}>

                {/* Expand / Collapse button */}
                <button
                  onClick={() => setPanelExpanded(p => !p)}
                  className="absolute top-3 right-3 z-10 text-white/70 hover:text-white transition-colors bg-white/10 hover:bg-white/20 rounded-lg p-1.5"
                  title={panelExpanded ? "Collapse" : "Expand"}
                >
                  <Maximize2 className="w-4 h-4" />
                </button>

                <AnimatePresence mode="wait">
                  {/* Empty State */}
                  {!isGenerating && !finalJobId && !draftJobId && (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center"
                    >
                      <div className="w-20 h-20 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center mb-4">
                        <Code2 className="w-10 h-10 text-white/20" />
                      </div>
                      <p className="text-lg font-light text-white/30">Generated notebook will appear here</p>
                    </motion.div>
                  )}

                  {/* Loading State */}
                  {isGenerating && (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 p-8 overflow-y-auto"
                    >
                      <div className="space-y-8">
                        {/* GIF animation */}
                        <div className="flex justify-center">
                          <img
                            src="/Scanning Documents.gif"
                            alt="Processing..."
                            style={{ width: 120, height: 120, objectFit: 'contain' }}
                          />
                        </div>

                        {/* Progress Steps */}
                        <div className="space-y-4 pl-4 border-l-2 border-white/5">
                          {steps.map((step, index) => {
                            const Icon = stepIcons[index]
                            const isActive = index + 1 === currentStep
                            const isCompleted = completedSteps[index]

                            return (
                              <motion.div
                                key={index}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{
                                  opacity: isActive || isCompleted ? 1 : 0.4,
                                  x: 0,
                                  scale: isActive ? 1.05 : 1,
                                }}
                                className="flex items-start gap-3"
                              >
                                <div className={cn(
                                  "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border-2 transition-all",
                                  isCompleted && "bg-white border-black",
                                  isActive && !isCompleted && "border-[#8ad4ff] bg-[#8ad4ff]/20",
                                  !isActive && !isCompleted && "border-white/10 bg-white/5"
                                )}>
                                  {isCompleted ? (
                                    <CheckCircle2 className="w-5 h-5 text-black" />
                                  ) : (
                                    <Icon className={cn("w-4 h-4", isActive ? "text-[#8ad4ff]" : "text-white/30")} />
                                  )}
                                </div>
                                <div>
                                  <h4 className={cn(
                                    "font-semibold text-sm transition-colors",
                                    isActive || isCompleted ? "text-white" : "text-white/40"
                                  )}>
                                    {step.name}
                                  </h4>
                                  {step.detail && (
                                    <p className="text-xs text-white/60 mt-1">{step.detail}</p>
                                  )}
                                </div>
                              </motion.div>
                            )
                          })}
                        </div>

                        {/* Stage 1 only: Live AI reading display (moved below activity for stage 2) */}
                        {currentStep === 1 && thinking && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-white/5 rounded-xl border border-white/10 overflow-hidden"
                          >
                            <button
                              onClick={() => setThinkingExpanded(p => !p)}
                              className="w-full flex items-center gap-2 px-4 py-3 bg-white/[0.03] hover:bg-white/[0.06] transition-colors"
                            >
                              <span className="relative flex h-2 w-2 shrink-0">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#8ad4ff] opacity-75" />
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#8ad4ff]" />
                              </span>
                              <span className="text-xs font-semibold text-white/80">AI is reading your paper...</span>
                              <ChevronDown className={`ml-auto w-3.5 h-3.5 text-white/40 transition-transform duration-200 ${thinkingExpanded ? 'rotate-180' : ''}`} />
                            </button>
                            {thinkingExpanded && (
                              <div
                                ref={thinkingScrollRef}
                                onScroll={handleThinkingScroll}
                                className="text-xs text-white/60 leading-relaxed p-4 max-h-48 overflow-y-auto border-t border-white/10"
                              >
                                <ReactMarkdown
                                  components={{
                                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                                    strong: ({ children }) => <strong className="text-white/90 font-semibold">{children}</strong>,
                                    em: ({ children }) => <em className="text-[#ffa8ff]/80">{children}</em>,
                                    h1: ({ children }) => <h1 className="text-sm font-bold text-white/90 mb-1">{children}</h1>,
                                    h2: ({ children }) => <h2 className="text-xs font-bold text-white/80 mb-1">{children}</h2>,
                                    h3: ({ children }) => <h3 className="text-xs font-semibold text-white/70 mb-1">{children}</h3>,
                                    ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
                                    ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
                                    li: ({ children }) => <li className="text-white/60">{children}</li>,
                                    code: ({ children }) => <code className="font-mono bg-white/10 px-1 rounded text-[#8ad4ff]">{children}</code>,
                                  }}
                                >
                                  {displayedThinking}
                                </ReactMarkdown>
                                <span className="inline-block w-1.5 h-3 bg-[#8ad4ff]/70 animate-pulse ml-0.5 align-middle" />
                              </div>
                            )}
                          </motion.div>
                        )}

                        {/* Stage 1 & 2: placeholder before thinking starts */}
                        {(currentStep === 1 || currentStep === 2) && !thinking && !activities.length && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-white/5 rounded-xl border border-white/10 overflow-hidden"
                          >
                            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-white/[0.03]">
                              <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#8ad4ff] opacity-75" />
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#8ad4ff]" />
                              </span>
                              <span className="text-xs font-semibold text-white/80">
                                {currentStep === 1 ? "AI is reading your paper..." : "AI is designing the implementation..."}
                              </span>
                            </div>
                            <div className="px-4 py-6 flex items-center gap-3">
                              <div className="flex gap-1">
                                {[0, 1, 2].map(i => (
                                  <motion.div
                                    key={i}
                                    className="w-1.5 h-1.5 rounded-full bg-white/30"
                                    animate={{ opacity: [0.3, 1, 0.3] }}
                                    transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                                  />
                                ))}
                              </div>
                              <span className="text-xs text-white/40">Uploading and parsing PDF...</span>
                            </div>
                          </motion.div>
                        )}

                        {/* arXiv metadata card — appears below thinking once thinking starts */}
                        {arxivMeta && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-white/5 rounded-xl border border-white/10 overflow-hidden"
                          >
                            <div className="flex items-start justify-between px-4 pt-3 pb-0">
                              <div className="flex flex-wrap gap-1.5 flex-1">
                                {arxivMeta.venue && (
                                  <span className="text-xs bg-[#ffd78a]/15 text-[#ffd78a] px-2 py-0.5 rounded-full font-medium">
                                    {arxivMeta.venue}
                                  </span>
                                )}
                                {arxivMeta.categories.slice(0, 3).map(cat => (
                                  <span key={cat} className="text-xs bg-[#8ad4ff]/15 text-[#8ad4ff] px-2 py-0.5 rounded-full font-medium">
                                    {CATEGORY_LABELS[cat] || cat}
                                  </span>
                                ))}
                                {arxivMeta.published && (
                                  <span className="text-xs bg-white/10 text-white/40 px-2 py-0.5 rounded-full">
                                    {arxivMeta.published}
                                  </span>
                                )}
                              </div>
                              <button
                                onClick={() => setShowArxivExpanded(true)}
                                className="ml-2 shrink-0 text-white/30 hover:text-white/70 transition-colors p-1"
                                title="Expand"
                              >
                                <Maximize2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                            <div className="px-4 py-3 space-y-1.5">
                              <h3 className="text-sm font-semibold text-white leading-snug">{arxivMeta.title}</h3>
                              <p className="text-xs text-white/40">{arxivMeta.authors}</p>
                              <p className="text-xs text-white/60 leading-relaxed line-clamp-3">{arxivMeta.abstract}</p>
                            </div>
                          </motion.div>
                        )}

                        {/* Early Download - available once draft is ready (stages 1-3 done) */}
                        {draftJobId && currentStep === 4 && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-[#ffd78a]/10 border border-[#ffd78a]/30 rounded-lg p-4"
                          >
                            <p className="text-xs text-[#ffd78a] font-semibold mb-2">Draft ready - download now while validation runs.</p>
                            <a
                              href={`${API_URL}/api/download/${draftJobId}`}
                              download
                              className="inline-flex items-center gap-2 bg-[#ffd78a]/20 hover:bg-[#ffd78a]/30 text-[#ffd78a] px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                            >
                              <Download className="w-4 h-4" />
                              Download Draft Notebook
                            </a>
                          </motion.div>
                        )}



                        {/* Stage 1 thinking (no activities yet) already rendered above */}

                        {/* Activity Feed */}
                        {activities.length > 0 && (
                          <div className="space-y-3">
                            {activities.map((activity, index) => (
                              <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="bg-white/5 rounded-lg p-4 border border-white/10"
                              >
                                <div className="text-xs text-[#8ad4ff] font-semibold uppercase tracking-wider mb-1">
                                  {activity.label}
                                </div>
                                <h4 className="text-sm font-semibold text-white mb-2">{activity.title}</h4>
                                {activity.type === 'analysis' && (
                                  <div className="space-y-2">
                                    {activity.content.authors && (
                                      <p className="text-xs text-white/40">{activity.content.authors}</p>
                                    )}
                                    {(activity.content.abstract_summary || activity.content.problem) && (
                                      <p className="text-xs text-white/60 leading-relaxed">
                                        {activity.content.abstract_summary || activity.content.problem}
                                      </p>
                                    )}
                                    {activity.content.insight && (
                                      <p className="text-xs text-[#8ad4ff]/70 italic">"{activity.content.insight}"</p>
                                    )}
                                    {/* Extracted tag rows */}
                                    {(activity.content.research_field || activity.content.key_contributions?.length || activity.content.algorithm_names?.length || activity.content.metrics?.length || activity.content.dataset_name || activity.content.key_layers?.length || activity.content.baseline_names?.length) && (
                                      <div className="mt-3 pt-3 border-t border-white/5 space-y-1.5">
                                        <TagRow label="Field"     tags={activity.content.research_field ? [activity.content.research_field] : []} color="violet" />
                                        <TagRow label="Goals"     tags={activity.content.key_contributions || []} color="pink" />
                                        <TagRow label="Methods"   tags={activity.content.algorithm_names || []} color="sky" />
                                        <TagRow label="Layers"    tags={activity.content.key_layers || []} color="teal" />
                                        <TagRow label="Baselines" tags={activity.content.baseline_names || []} color="rose" />
                                        <TagRow label="Metrics"   tags={activity.content.metrics || []} color="emerald" />
                                        <TagRow label="Data"      tags={activity.content.dataset_name ? [activity.content.dataset_name] : []} color="amber" />
                                      </div>
                                    )}
                                  </div>
                                )}
                                {activity.type === 'design' && (
                                  <div className="mt-2 pt-2 border-t border-white/5 space-y-1.5">
                                    <TagRow label="Arch"   tags={activity.content.model_type ? [activity.content.model_type] : []} color="sky" />
                                    <TagRow label="Dim"    tags={activity.content.embed_dim ? [`dim=${activity.content.embed_dim}`] : []} color="teal" />
                                    <TagRow label="Layers" tags={activity.content.num_layers ? [`${activity.content.num_layers} layers`] : []} color="violet" />
                                    <TagRow label="Heads"  tags={activity.content.num_heads ? [`${activity.content.num_heads} heads`] : []} color="pink" />
                                  </div>
                                )}
                              </motion.div>
                            ))}
                          </div>
                        )}

                        {/* Stage 2: AI designing — appears below activity cards */}
                        {currentStep === 2 && thinking && (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-white/5 rounded-xl border border-white/10 overflow-hidden"
                          >
                            <button
                              onClick={() => setThinkingExpanded(p => !p)}
                              className="w-full flex items-center gap-2 px-4 py-3 bg-white/[0.03] hover:bg-white/[0.06] transition-colors"
                            >
                              <span className="relative flex h-2 w-2 shrink-0">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#8ad4ff] opacity-75" />
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#8ad4ff]" />
                              </span>
                              <span className="text-xs font-semibold text-white/80">AI is designing the implementation...</span>
                              <ChevronDown className={`ml-auto w-3.5 h-3.5 text-white/40 transition-transform duration-200 ${thinkingExpanded ? 'rotate-180' : ''}`} />
                            </button>
                            {thinkingExpanded && (
                            <div
                              ref={thinkingScrollRef}
                              onScroll={handleThinkingScroll}
                              className="text-xs text-white/60 leading-relaxed p-4 max-h-48 overflow-y-auto border-t border-white/10"
                            >
                              <ReactMarkdown
                                components={{
                                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                                  strong: ({ children }) => <strong className="text-white/90 font-semibold">{children}</strong>,
                                  em: ({ children }) => <em className="text-[#ffa8ff]/80">{children}</em>,
                                  h1: ({ children }) => <h1 className="text-sm font-bold text-white/90 mb-1">{children}</h1>,
                                  h2: ({ children }) => <h2 className="text-xs font-bold text-white/80 mb-1">{children}</h2>,
                                  h3: ({ children }) => <h3 className="text-xs font-semibold text-white/70 mb-1">{children}</h3>,
                                  ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
                                  ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
                                  li: ({ children }) => <li className="text-white/60">{children}</li>,
                                  code: ({ children }) => <code className="font-mono bg-white/10 px-1 rounded text-[#8ad4ff]">{children}</code>,
                                }}
                              >
                                {displayedThinking}
                              </ReactMarkdown>
                              <span className="inline-block w-1.5 h-3 bg-[#8ad4ff]/70 animate-pulse ml-0.5 align-middle" />
                            </div>
                            )}
                          </motion.div>
                        )}
                      </div>
                    </motion.div>
                  )}

                  {/* Success State */}
                  {showDone && (
                    <motion.div
                      key="done"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center"
                    >
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2, type: "spring" }}
                        className="w-20 h-20 rounded-full bg-[#8ad4ff]/20 flex items-center justify-center border-2 border-[#8ad4ff] mb-6"
                      >
                        <CheckCircle2 className="w-10 h-10 text-[#8ad4ff]" />
                      </motion.div>
                      <h2 className="text-2xl font-bold text-white mb-2">Notebook Ready!</h2>
                      <p className="text-white/60 mb-6">All cells validated successfully</p>

                      {/* Download Buttons */}
                      {finalJobId && (
                        <div className="flex flex-col sm:flex-row gap-3 w-full">
                          <a
                            href={`${API_URL}/api/download/${finalJobId}`}
                            download
                            className="flex items-center justify-center gap-2 bg-white hover:bg-gray-100 text-black px-6 py-3 rounded-lg font-semibold transition-all active:scale-95 shadow-lg border-2 border-black flex-1"
                          >
                            <Download className="w-4 h-4" />
                            Download Notebook
                          </a>
                          <button
                            onClick={async () => {
                              try {
                                // Create GitHub Gist via backend (uses GitHub token)
                                const gistResponse = await fetch(`${API_URL}/api/create-gist/${finalJobId}`, {
                                  method: 'POST',
                                });

                                if (!gistResponse.ok) {
                                  const errorText = await gistResponse.text();
                                  throw new Error(`Failed to create Gist: ${errorText}`);
                                }

                                const gistData = await gistResponse.json();

                                // Open in Google Colab with the proper Colab URL
                                window.open(gistData.colab_url, '_blank');
                              } catch (err: any) {
                                console.error('Failed to open in Colab:', err);
                                alert(`Failed to open in Colab: ${err.message}`);
                              }
                            }}
                            className="flex items-center justify-center gap-2 bg-white hover:bg-gray-100 text-black px-6 py-3 rounded-lg font-semibold transition-all active:scale-95 shadow-lg border-2 border-black flex-1"
                          >
                            <img src="/icons8-google-colab.svg" alt="Colab" className="w-5 h-5" />
                            Open in Colab
                          </button>
                        </div>
                      )}
                    </motion.div>
                  )}

                  {/* Error State */}
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="absolute bottom-4 left-4 right-4 bg-red-500/10 border border-red-500/50 rounded-lg p-4 backdrop-blur-sm"
                    >
                      <p className="text-sm text-red-300">{error}</p>
                    </motion.div>
                  )}
                </AnimatePresence>

              </div>
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-center mt-8 text-xs text-white/40"
        >
          Powered by OpenRouter · Real PyTorch implementations · Bring your own API key
        </motion.div>

      </div>

      {/* Trending Papers — full width */}
      <motion.section
          ref={trendingSectionRef}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-14 px-8 pb-16 relative z-10"
        >
          {/* Header + period tabs */}
          <div className="flex items-center gap-3 mb-6 flex-wrap">
            <h2 className="text-lg font-bold text-white tracking-tight">Trending Papers</h2>
            <span className="text-[11px] bg-[#8ad4ff]/10 text-[#8ad4ff] border border-[#8ad4ff]/20 px-2.5 py-0.5 rounded-full font-medium">
              HuggingFace
            </span>
            <div className="ml-auto flex items-center gap-2">
              {/* Sort by stars toggle */}
              <div className="relative group/tip">
                <button
                  onClick={() => setTrendingSort(s => s === 'stars' ? 'new' : 'stars')}
                  className={`p-2 rounded-lg transition-all ${
                    trendingSort === 'stars'
                      ? 'bg-[#8ad4ff]/20 border border-[#8ad4ff]/30'
                      : 'bg-white/5 hover:bg-white/10 border border-transparent'
                  }`}
                >
                  <svg className={`w-4 h-4 transition-colors ${trendingSort === 'stars' ? 'text-[#8ad4ff]' : 'text-white/40'}`} viewBox="0 0 24 24" fill={trendingSort === 'stars' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                  </svg>
                </button>
                <div className="absolute right-0 top-full mt-2 w-56 bg-[#111] border border-white/10 rounded-xl px-3 py-2.5 text-xs text-white/70 leading-relaxed shadow-xl z-20 pointer-events-none opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150">
                  {trendingSort === 'stars' ? 'Sorted by GitHub stars — click for newest first' : 'Sorted by newest — click for GitHub stars'}
                  <div className="absolute -top-1.5 right-3 w-3 h-3 bg-[#111] border-l border-t border-white/10 rotate-45" />
                </div>
              </div>
              {/* Period tabs */}
              <div className="flex items-center gap-1 bg-white/5 rounded-xl p-1">
                {(['day', 'week', 'month'] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setTrendingPeriod(p)}
                    className={`text-xs font-medium px-3.5 py-1.5 rounded-lg transition-all ${
                      trendingPeriod === p
                        ? 'bg-[#8ad4ff]/20 text-[#8ad4ff] border border-[#8ad4ff]/30'
                        : 'text-white/40 hover:text-white/70'
                    }`}
                  >
                    {p === 'day' ? 'Daily' : p === 'week' ? 'Weekly' : 'Monthly'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {trendingLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from({ length: 8 }).map((_, i) => <PaperCardSkeleton key={i} />)}
            </div>
          ) : trendingPapers.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...trendingPapers]
                .sort((a, b) =>
                  trendingSort === 'stars'
                    ? (b.githubStars - a.githubStars) || (b.upvotes - a.upvotes)
                    : (b.publishedAt > a.publishedAt ? 1 : -1)
                )
                .map(paper => (
                <PaperCard key={paper.id} paper={paper} onConvert={handleConvertPaper} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-white/30 text-center py-10">No papers found for this period.</p>
          )}
      </motion.section>

      {/* Fullscreen Thinking Modal */}
      <AnimatePresence>
        {/* arXiv expanded modal */}
        {showArxivExpanded && arxivMeta && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            onClick={() => setShowArxivExpanded(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-[#0a0a0a] border border-white/10 rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6 space-y-4"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex flex-wrap gap-1.5">
                  {arxivMeta.venue && (
                    <span className="text-xs bg-[#ffd78a]/15 text-[#ffd78a] px-2 py-0.5 rounded-full font-medium">
                      {arxivMeta.venue}
                    </span>
                  )}
                  {arxivMeta.categories.slice(0, 4).map(cat => (
                    <span key={cat} className="text-xs bg-[#8ad4ff]/15 text-[#8ad4ff] px-2 py-0.5 rounded-full font-medium">
                      {CATEGORY_LABELS[cat] || cat}
                    </span>
                  ))}
                  {arxivMeta.published && (
                    <span className="text-xs bg-white/10 text-white/40 px-2 py-0.5 rounded-full">
                      {arxivMeta.published}
                    </span>
                  )}
                </div>
                <button onClick={() => setShowArxivExpanded(false)} className="text-white/40 hover:text-white shrink-0">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <h2 className="text-lg font-bold text-white leading-snug">{arxivMeta.title}</h2>
              <p className="text-sm text-white/50">{arxivMeta.authors}</p>
              <p className="text-sm text-white/70 leading-relaxed">{arxivMeta.abstract}</p>
            </motion.div>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  )
}
