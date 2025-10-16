"""
RAG-based Flask Application with OOP Design
Clean architecture using design patterns
"""
import os
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
            print(f"\n❌ PDF processing failed: {e}")
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


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get current model configuration"""
    try:
        emb_config = config.get_embedding_model()
        llm_config = config.get_llm_model()
        
        return jsonify({
            'embedding_model': emb_config.display_name,
            'llm_model': llm_config.display_name,
            'embedding_quality': emb_config.quality_score,
            'llm_quality': llm_config.quality_score,
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

