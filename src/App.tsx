/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Eraser, Brain, Activity, RotateCcw, AlertCircle } from 'lucide-react';

// Design Constants from Recipe 3: Hardware / Specialist Tool
const THEME = {
  bg: '#151619',
  card: '#1c1d21',
  accent: '#4FD1C5', // Teal accent
  textPrimary: '#FFFFFF',
  textSecondary: '#8E9299',
  mono: '"JetBrains Mono", "Roboto Mono", monospace',
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<{ result: string; confidence: number } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Canvas Setup
  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Standard MNIST is 28x28, but we'll draw on a larger canvas and downscale
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  useEffect(() => {
    setupCanvas();
  }, [setupCanvas]);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.beginPath();
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setError(null);
  };

  const preprocessAndPredict = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    setError(null);

    try {
      // 1. Downscale to 28x28
      const smallCanvas = document.createElement('canvas');
      smallCanvas.width = 28;
      smallCanvas.height = 28;
      const smallCtx = smallCanvas.getContext('2d');
      if (!smallCtx) throw new Error("Could not init small canvas");

      smallCtx.drawImage(canvas, 0, 0, 28, 28);
      
      // 2. Extract grayscale values
      const imageData = smallCtx.getImageData(0, 0, 28, 28);
      const pixels: number[] = [];
      for (let i = 0; i < imageData.data.length; i += 4) {
        // Just take the red channel since it's white on black
        pixels.push(imageData.data[i] / 255.0);
      }

      // 3. Call API
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: pixels }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Prediction failed. Is the server running?');
      }

      const data = await response.json();
      setPrediction({ result: data.prediction, confidence: data.confidence });
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center justify-center font-sans" style={{ backgroundColor: THEME.bg, color: THEME.textPrimary }}>
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-12 text-center"
      >
        <span className="text-xs tracking-[0.2em] font-medium opacity-50 uppercase mb-2 block" style={{ fontFamily: THEME.mono }}>
          Neural Network Interface // v1.0.4
        </span>
        <h1 className="text-5xl font-light tracking-tight mb-2">Finger-Scribe</h1>
        <p className="text-sm opacity-60 max-w-md mx-auto">
          Draw a single digit (0-9) inside the grid for real-time scikit-learn classification.
        </p>
      </motion.div>

      <div className="flex flex-col lg:flex-row gap-12 items-start justify-center">
        {/* Drawing Area */}
        <div className="relative">
          <div 
            className="rounded-2xl overflow-hidden border-2 border-white/5 shadow-2xl relative"
            style={{ width: 320, height: 320, backgroundColor: 'black' }}
          >
            {/* Grid Overlay */}
            <div className="absolute inset-0 pointer-events-none opacity-10" 
                 style={{ 
                   backgroundImage: `linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)`,
                   backgroundSize: '32px 32px'
                 }} 
            />
            
            <canvas
              ref={canvasRef}
              width={320}
              height={320}
              onMouseDown={startDrawing}
              onMouseUp={stopDrawing}
              onMouseOut={stopDrawing}
              onMouseMove={draw}
              onTouchStart={startDrawing}
              onTouchEnd={stopDrawing}
              onTouchMove={draw}
              className="cursor-crosshair block touch-none"
            />
          </div>

          {/* Controls */}
          <div className="flex gap-4 mt-6 justify-center">
            <button 
              onClick={clearCanvas}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm font-medium border border-white/10"
            >
              <Eraser size={16} />
              Reset
            </button>
            <button 
              onClick={preprocessAndPredict}
              disabled={isLoading}
              className="flex items-center gap-2 px-6 py-2 rounded-lg transition-all text-sm font-bold shadow-lg disabled:opacity-50"
              style={{ backgroundColor: THEME.accent, color: '#000' }}
            >
              {isLoading ? (
                <RotateCcw className="animate-spin" size={16} />
              ) : (
                <Brain size={18} />
              )}
              {isLoading ? 'Processing...' : 'Identify'}
            </button>
          </div>
        </div>

        {/* Results Pane */}
        <div className="w-full lg:w-80 space-y-6">
          <div className="p-6 rounded-2xl border border-white/5" style={{ backgroundColor: THEME.card }}>
            <div className="flex items-center justify-between mb-6">
              <span className="text-[10px] uppercase tracking-widest font-bold opacity-40" style={{ fontFamily: THEME.mono }}>
                Classification Output
              </span>
              <Activity size={14} className="opacity-40" />
            </div>

            <AnimatePresence mode="wait">
              {prediction ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="text-center"
                >
                  <div className="text-[120px] font-light leading-none mb-4" style={{ fontFamily: THEME.mono, color: THEME.accent }}>
                    {prediction.result}
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-[10px] uppercase font-bold opacity-40" style={{ fontFamily: THEME.mono }}>
                      <span>Confidence</span>
                      <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence * 100}%` }}
                        className="h-full" 
                        style={{ backgroundColor: THEME.accent }} 
                      />
                    </div>
                  </div>
                </motion.div>
              ) : error ? (
                <motion.div 
                  key="error"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center gap-3 py-6"
                >
                  <AlertCircle size={32} className="text-red-400 opacity-60" />
                  <p className="text-xs text-red-300 text-center leading-relaxed">
                    {error}
                  </p>
                </motion.div>
              ) : (
                <motion.div 
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="py-12 text-center opacity-30 italic text-sm"
                >
                  Waiting for input...
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Model Info */}
          <div className="p-6 rounded-2xl border border-white/5" style={{ backgroundColor: THEME.card }}>
             <span className="text-[10px] uppercase tracking-widest font-bold opacity-40 block mb-4" style={{ fontFamily: THEME.mono }}>
                System Stats
              </span>
              <div className="space-y-3 font-mono text-[11px] opacity-70">
                <div className="flex justify-between">
                  <span>Engine:</span>
                  <span>Scikit-Learn MLP</span>
                </div>
                <div className="flex justify-between">
                  <span>Input:</span>
                  <span>28x28 (Flat)</span>
                </div>
                <div className="flex justify-between">
                  <span>Layers:</span>
                  <span>128, 64</span>
                </div>
              </div>
          </div>
        </div>
      </div>

      {/* Background aesthetic circles */}
      <div className="fixed top-1/4 -right-20 w-80 h-80 rounded-full blur-[120px] pointer-events-none opacity-10" style={{ backgroundColor: THEME.accent }} />
      <div className="fixed bottom-1/4 -left-20 w-80 h-80 rounded-full blur-[120px] pointer-events-none opacity-5" style={{ backgroundColor: '#ff4e00' }} />
    </div>
  );
}
