import React, { useState, useEffect } from 'react'
import MarketGlobe from './components/MarketGlobe'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Systems online. I am your Agentic Financial Assistant. How can I help you navigate the markets today?' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [stockData, setStockData] = useState(null)

  const [isListening, setIsListening] = useState(false)

  // Voice Interaction (STT)
  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return alert("Browser does not support voice recognition.")
    
    const recognition = new SpeechRecognition()
    recognition.onstart = () => setIsListening(true)
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setInput(transcript)
      setIsListening(false)
    }
    recognition.onerror = () => setIsListening(false)
    recognition.start()
  }

  // Voice Synthesis (TTS)
  const speak = (text) => {
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 1.1
    utterance.pitch = 0.9
    window.speechSynthesis.speak(utterance)
  }

  const handleSend = async () => {
    if (!input.trim()) return
    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    const currentInput = input
    setInput('')
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/chat`, { prompt: currentInput })
      const aiContent = response.data.content
      setMessages(prev => [...prev, { role: 'assistant', content: aiContent }])
      speak(aiContent) // Speak the response
    } catch (err) {
      console.error(err)
      setMessages(prev => [...prev, { role: 'assistant', content: "Error communicating with intelligence core." }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden font-sans text-white">
      {/* 3D Background */}
      <MarketGlobe />

      {/* Main UI Overlay */}
      <div className="relative z-10 grid grid-cols-12 gap-6 p-8 h-full pb-20">
        
        {/* Left Sidebar: Market Stats */}
        <motion.div 
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="col-span-3 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-2xl"
        >
          <h2 className="text-2xl font-bold mb-6 text-cyan-400">Market Pulse</h2>
          <div className="space-y-4">
            {['NVDA', 'AAPL', 'BTC-USD', 'TSLA'].map(sym => (
              <div key={sym} className="p-4 bg-white/5 rounded-2xl border border-white/5 hover:border-cyan-500/50 transition-all cursor-pointer">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-lg">{sym}</span>
                  <span className="text-green-400 font-bold">+2.4%</span>
                </div>
                <div className="h-1 bg-white/10 mt-2 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: '70%' }}
                    className="h-full bg-cyan-400"
                  />
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-8 pt-8 border-t border-white/10">
            <h3 className="text-sm uppercase tracking-widest text-slate-400 mb-4">Active Agents</h3>
            <div className="flex gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" title="Finance Agent Active" />
              <div className="w-3 h-3 rounded-full bg-cyan-500 animate-pulse" title="Web Agent Active" />
              <div className="w-3 h-3 rounded-full bg-purple-500" title="Sentiment Agent Idle" />
            </div>
          </div>
        </motion.div>

        {/* Center: Chat & Analysis */}
        <div className="col-span-6 flex flex-col h-full space-y-6">
          {/* Global News Ticker (Static Demo) */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-4 overflow-hidden h-24 relative">
             <div className="absolute top-0 left-0 bg-cyan-500/20 px-2 py-0.5 text-[10px] uppercase tracking-tighter rounded-br-lg z-20">Live Intelligence Feed</div>
             <motion.div 
               animate={{ y: [0, -100] }}
               transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
               className="space-y-2 mt-4"
             >
               {[
                 "FED indicates potential rate pause in Q3...",
                 "Nvidia reports record data center revenue...",
                 "Bitcoin breaks resistance at $65k level...",
                 "Apple announces new AI integration in iOS..."
               ].map((news, i) => (
                 <div key={i} className="text-xs text-slate-300 border-l-2 border-cyan-500/30 pl-3">{news}</div>
               ))}
             </motion.div>
          </div>

          {/* Chat Display */}
          <div className="flex-grow bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl p-6 overflow-y-auto custom-scrollbar flex flex-col gap-4">
            <AnimatePresence>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[85%] p-5 rounded-3xl shadow-xl ${
                    msg.role === 'user' 
                      ? 'bg-gradient-to-br from-cyan-600/30 to-blue-600/20 border border-cyan-500/30 text-white' 
                      : 'bg-white/5 border border-white/10 text-slate-200'
                  }`}>
                    {/* Render reasoning steps if they exist in the message */}
                    {msg.content.includes("Step") ? (
                      <div className="space-y-4">
                        {msg.content.split("\n").map((line, li) => (
                          line.startsWith("Step") ? (
                            <div key={li} className="flex items-center gap-3 text-cyan-400 font-mono text-xs uppercase tracking-widest bg-cyan-500/10 p-2 rounded-lg border border-cyan-500/20">
                              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                              {line}
                            </div>
                          ) : <div key={li} className="pl-6">{line}</div>
                        ))}
                      </div>
                    ) : msg.content}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {loading && (
              <div className="flex items-center gap-3 text-cyan-500 font-mono text-xs italic p-4 bg-white/5 rounded-2xl border border-white/5">
                <div className="flex gap-1">
                  <div className="w-1 h-1 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '0s'}} />
                  <div className="w-1 h-1 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}} />
                  <div className="w-1 h-1 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '0.4s'}} />
                </div>
                Syncing with Agentic Collective...
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-full p-2 flex items-center pr-4 shadow-2xl">
            <button 
              onClick={startListening}
              className={`p-3 rounded-full transition-colors ${isListening ? 'bg-red-500 animate-pulse' : 'hover:bg-white/10'}`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            </button>
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask by voice or text..."
              className="flex-grow bg-transparent border-none outline-none px-6 text-lg placeholder:text-slate-500"
            />
            <button 
              onClick={handleSend}
              className="bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-bold px-6 py-2 rounded-full transition-colors"
            >
              SEND
            </button>
          </div>
        </div>

        {/* Right Sidebar: Sentiment & Visuals */}
        <motion.div 
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="col-span-3 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-2xl overflow-y-auto custom-scrollbar"
        >
          <h2 className="text-2xl font-bold mb-6 text-purple-400">Agent Intelligence</h2>
          
          <div className="bg-white/5 rounded-2xl p-4 border border-white/5 mb-6">
            <h3 className="text-xs text-slate-400 uppercase mb-2">Global Sentiment</h3>
            <div className="text-3xl font-bold text-center py-4 drop-shadow-[0_0_15px_rgba(168,85,247,0.5)]">7.8 / 10</div>
            <div className="text-center text-sm text-green-400">Bullish Momentum</div>
          </div>

          <div className="bg-cyan-500/10 rounded-2xl p-4 border border-cyan-500/20 mb-6">
            <h2 className="text-lg font-bold mb-4 text-cyan-400 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
              Simulated Portfolio
            </h2>
            <div className="space-y-3">
              {[
                { sym: 'NVDA', qty: 10, val: '$8,900' },
                { sym: 'AAPL', qty: 5, val: '$911' },
                { sym: 'BTC', qty: 0.5, val: '$32,115' }
              ].map(item => (
                <div key={item.sym} className="flex justify-between text-sm">
                  <span className="text-slate-400">{item.sym} ({item.qty})</span>
                  <span className="font-mono text-cyan-300">{item.val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-4">
             <h3 className="text-xs text-slate-400 uppercase">Live Operations</h3>
             <ul className="text-sm space-y-3">
               {[
                 "Analyzing NVDA Q4 report...",
                 "Scanning for sentiment spikes...",
                 "Updating portfolio balances..."
               ].map((task, i) => (
                 <li key={i} className="flex items-center gap-3 group">
                   <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 group-hover:animate-ping" />
                   <span className="text-slate-300 group-hover:text-white transition-colors">{task}</span>
                 </li>
               ))}
             </ul>
          </div>
        </motion.div>

      </div>

      {/* 3D Market Ticker */}
      <div className="absolute bottom-0 w-full bg-cyan-500/10 backdrop-blur-lg border-t border-cyan-500/20 h-16 flex items-center overflow-hidden">
        <motion.div 
          animate={{ x: ['100%', '-100%'] }}
          transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
          className="flex whitespace-nowrap gap-12 text-lg font-mono"
        >
          {['NVDA: $890.43 (+2.1%)', 'AAPL: $182.31 (-0.5%)', 'BTC: $64,231 (+1.2%)', 'TSLA: $175.05 (+0.8%)', 'ETH: $3,421 (+0.5%)'].map(item => (
            <span key={item} className="text-cyan-300 drop-shadow-[0_0_8px_rgba(0,242,255,0.5)]">
              {item}
            </span>
          ))}
        </motion.div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
      `}</style>
    </div>
  )
}
