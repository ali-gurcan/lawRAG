"""
RAG-based Flask Application with OOP Design
Clean architecture using design patterns
"""
import os
# M4 Mac OpenMP fix
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# Import OOP components
from core import get_config
from services import RAGEngine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Banner
print("\n" + "="*70)
print("🚀 RAG SYSTEM - Enhanced OOP Architecture")
print("="*70)

# Load configuration
print("\n[1/5] Loading configuration...")
config = get_config('config.yaml')

# Validate configuration
if not config.validate():
    print("❌ Configuration validation failed!")
    exit(1)

# Print configuration summary
config.print_summary()

# Initialize RAG Engine
print("[2/5] Initializing RAG Engine...")
try:
    rag_engine = RAGEngine(config)
    print("✅ RAG Engine initialized successfully!\n")
except Exception as e:
    print(f"❌ Failed to initialize RAG Engine: {e}")
    exit(1)

# Process PDFs
print("[3/5] Scanning for PDF documents...")
docs_dir = config.docs_dir

if os.path.exists(docs_dir):
    pdf_files = [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.endswith('.pdf')
    ]
    
    if pdf_files:
        print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"   {i}. {os.path.basename(pdf)}")
        
        print("\n[4/5] Processing PDFs and building vector database...")
        print("⏳ This may take a few minutes on first run...")
        print("💡 Subsequent runs will use cache (fast!)\n")
        
        try:
            rag_engine.process_pdfs(pdf_files)
            print("\n✅ PDF processing complete!")
        except Exception as e:
            import traceback
            print(f"\n❌ PDF processing failed: {e}")
            print("\n📋 Full traceback:")
            traceback.print_exc()
            exit(1)
    else:
        print(f"⚠️  No PDF files found in {docs_dir}/")
        print(f"📁 Please add PDF files to continue")
        exit(1)
else:
    print(f"❌ Directory {docs_dir}/ not found!")
    print(f"📁 Creating directory...")
    os.makedirs(docs_dir, exist_ok=True)
    print(f"⚠️  Please add PDF files to {docs_dir}/ and restart")
    exit(1)

print("\n[5/5] Starting Flask server...")
print("="*70)
print(f"✅ Server ready at http://localhost:{config.server.port}")
print("="*70 + "\n")


# Routes
@app.route('/')
def index():
    """Main page - ChatGPT Style RAG UI"""
    return render_template('index_chatgpt.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Non-streaming chat endpoint (fallback)"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if not rag_engine.is_ready():
            return jsonify({'error': 'RAG system not ready'}), 503
        
        # Generate answer
        result = rag_engine.generate_answer(user_message)
        
        return jsonify({
            'response': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'num_sources': result['num_sources'],
            'mode': 'RAG',
            'low_confidence': result.get('low_confidence', False),
            'warning': result.get('warning', None)
        })
    
    except Exception as e:
        print(f"❌ Error in /api/chat: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint (Server-Sent Events)"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if not rag_engine.is_ready():
            return jsonify({'error': 'RAG system not ready'}), 503
        
        # Generate streaming response
        def generate():
            try:
                for chunk in rag_engine.generate_answer_stream(user_message):
                    yield chunk
            except Exception as e:
                import json
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
    
    except Exception as e:
        print(f"❌ Error in /api/chat/stream: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _detect_quantization_info(llm_backend: str, llm_config) -> tuple:
    """
    Detect quantization information dynamically from environment and model state
    
    Returns:
        tuple: (quantization_info, model_path)
    """
    # Quantization patterns (can be extended)
    quantization_patterns = {
        'q4_k_m': ('Q4_K_M', '4-bit'),
        'q4_k_s': ('Q4_K_S', '4-bit'),
        'q5_k_m': ('Q5_K_M', '5-bit'),
        'q8_0': ('Q8_0', '8-bit'),
        'q8': ('Q8', '8-bit'),
        'f16': ('F16', '16-bit'),
        'f32': ('F32', '32-bit'),
        'nf4': ('NF4', '4-bit'),
        'q4': ('Q4', '4-bit'),
        'q5': ('Q5', '5-bit')
    }
    
    # Backend types
    BACKEND_OLLAMA = 'ollama'
    BACKEND_LLAMA_CPP_VARIANTS = ('llama_cpp', 'llamacpp', 'llama-cpp')
    BACKEND_TRANSFORMERS = 'transformers'
    
    backend_lower = llm_backend.lower().strip()
    
    # Detect from backend type
    if backend_lower == BACKEND_OLLAMA:
        model_name = os.getenv('OLLAMA_MODEL', '')
        model_path = f"Ollama: {model_name}" if model_name else "Ollama: Not specified"
        
        # Detect quantization from model name
        model_lower = model_name.lower()
        for pattern, (name, bits) in quantization_patterns.items():
            if pattern in model_lower:
                return f"{name} ({bits} via Ollama)", model_path
        
        # Check for precision indicators
        if any(p in model_lower for p in ['f16', 'f32', 'fp16', 'fp32']):
            return "Full Precision (F16/F32 via Ollama)", model_path
        
        return "Ollama default quantization", model_path
    
    elif backend_lower in BACKEND_LLAMA_CPP_VARIANTS:
        model_path = os.getenv('LLM_GGUF_PATH', '')
        if model_path and os.path.exists(model_path):
            filename_lower = os.path.basename(model_path).lower()
            
            # Detect from filename
            for pattern, (name, bits) in quantization_patterns.items():
                if pattern in filename_lower:
                    return f"{name} ({bits} GGUF)", model_path
            
            return "GGUF (detected from filename)", model_path
        else:
            return "GGUF (path not set)", model_path or "N/A"
    
    else:  # Transformers backend
        hf_quant = os.getenv('LLM_QUANT', '').lower()
        model_path = getattr(llm_config, 'name', 'Unknown')
        
        # Check quantization env var
        for pattern, (name, bits) in quantization_patterns.items():
            if pattern in hf_quant:
                return f"{name} ({bits} via bitsandbytes)", model_path
        
        # Check model config for dtype
        dtype = getattr(llm_config, 'dtype', 'float32')
        if 'float16' in dtype.lower() or 'fp16' in dtype.lower():
            return "FP16 (16-bit)", model_path
        elif 'bfloat16' in dtype.lower() or 'bf16' in dtype.lower():
            return "BF16 (16-bit)", model_path
        
        return f"Full Precision ({dtype.upper()})", model_path


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get current model configuration with backend and quantization info"""
    try:
        emb_config = config.get_embedding_model()
        llm_config = config.get_llm_model()
        
        # Detect LLM backend from environment
        llm_backend = os.getenv('LLM_BACKEND', 'transformers').lower().strip()
        
        # Get quantization info dynamically
        quantization_info, model_path = _detect_quantization_info(
            llm_backend, 
            llm_config
        )
        
        return jsonify({
            'embedding_model': {
                'name': emb_config.display_name,
                'model_id': emb_config.name,
                'quality': emb_config.quality_score,
                'dimension': emb_config.dimension,
                'size_gb': emb_config.size_gb
            },
            'llm_model': {
                'name': llm_config.display_name,
                'model_id': llm_config.name,
                'quality': llm_config.quality_score,
                'size_gb': llm_config.size_gb,
                'backend': llm_backend,
                'quantization': quantization_info,
                'model_path': model_path,
                'device': getattr(llm_config, 'device', 'cpu'),
                'dtype': getattr(llm_config, 'dtype', 'float32')
            },
            'reranker': {
                'name': config.get_reranker_model().display_name,
                'model_id': config.get_reranker_model().name
            },
            'features': {
                'hybrid_search': config.retrieval.use_hybrid_search,
                'reranking': config.retrieval.use_reranking,
                'query_expansion': config.retrieval.use_query_expansion,
                'streaming': config.streaming.enable_streaming
            }
        })
    except Exception as e:
        print(f"❌ Error in /api/models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config_api():
    """Get full configuration (for admin)"""
    try:
        from dataclasses import asdict
        
        return jsonify({
            'retrieval': asdict(config.retrieval),
            'chunking': asdict(config.chunking),
            'confidence': asdict(config.confidence),
            'streaming': asdict(config.streaming),
            'active_models': {
                'embedding': config.active_embedding_model,
                'llm': config.active_llm_model,
                'reranker': config.active_reranker_model
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_ready': rag_engine.is_ready(),
        'models_loaded': True
    })


if __name__ == '__main__':
    # Run server with configuration
    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug,
        threaded=config.server.threaded
    )

