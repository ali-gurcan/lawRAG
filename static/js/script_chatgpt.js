// State
let isWaitingForResponse = false;
let messageHistory = [];

// Markdown render function
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    // Fallback: simple markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*$)/gm, '<li>$1. $2</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/^(.*)$/gm, '<p>$1</p>')
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<h[12]>.*<\/h[12]>)<\/p>/g, '$1')
        .replace(/<p>(<li>.*<\/li>)<\/p>/g, '<ul>$1</ul>')
        .replace(/<\/li><li>/g, '</li><li>');
}

// DOM Elements
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const charCount = document.getElementById('charCount');
const systemStatus = document.getElementById('systemStatus');
const chunkCount = document.getElementById('chunkCount');
const pdfList = document.getElementById('pdfList');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
    loadPDFList();
    loadModelInfo();
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
        charCount.textContent = messageInput.value.length;
    });
    
    // Send on Enter (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        // Handle new API structure (nested objects)
        if (data.embedding_model && data.llm_model) {
            // Extract names from nested structure
            const embeddingInfo = typeof data.embedding_model === 'string' 
                ? { name: data.embedding_model } 
                : data.embedding_model;
            
            const llmInfo = typeof data.llm_model === 'string'
                ? { name: data.llm_model }
                : data.llm_model;
            
            // Get display names
            const embeddingName = embeddingInfo.name || embeddingInfo.display_name || 'Bilinmiyor';
            const llmName = llmInfo.name || llmInfo.display_name || 'Bilinmiyor';
            
            // Update embedding badge
            const embeddingElement = document.getElementById('embeddingModel');
            if (embeddingElement) {
                embeddingElement.textContent = embeddingName;
                embeddingElement.title = embeddingInfo.model_id || embeddingName;
            }
            
            // Update LLM badge with quantization info if available
            const llmElement = document.getElementById('llmModel');
            if (llmElement) {
                let displayText = llmName;
                if (llmInfo.quantization) {
                    displayText += ` (${llmInfo.quantization.split(' ')[0]})`;
                }
                llmElement.textContent = displayText;
                
                // Tooltip with full info
                const tooltipParts = [
                    `Model: ${llmName}`,
                    `Backend: ${llmInfo.backend || 'transformers'}`,
                ];
                if (llmInfo.quantization) {
                    tooltipParts.push(`Quantization: ${llmInfo.quantization}`);
                }
                if (llmInfo.model_path) {
                    tooltipParts.push(`Path: ${llmInfo.model_path}`);
                }
                llmElement.title = tooltipParts.join('\n');
            }
        } else {
            throw new Error('Invalid API response structure');
        }
    } catch (error) {
        console.error('Model bilgisi yüklenemedi:', error);
        const embeddingEl = document.getElementById('embeddingModel');
        const llmEl = document.getElementById('llmModel');
        if (embeddingEl) embeddingEl.textContent = 'Yüklenemedi';
        if (llmEl) llmEl.textContent = 'Yüklenemedi';
    }
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.rag_ready) {
            systemStatus.querySelector('span').textContent = 'RAG Sistemi Hazır';
            systemStatus.querySelector('.status-dot').style.background = '#10a37f';
        } else {
            systemStatus.querySelector('span').textContent = 'Yükleniyor...';
            systemStatus.querySelector('.status-dot').style.background = '#f59e0b';
            setTimeout(loadSystemStatus, 3000);
        }
    } catch (error) {
        systemStatus.querySelector('span').textContent = 'Bağlantı Hatası';
        systemStatus.querySelector('.status-dot').style.background = '#ef4444';
    }
}

// Load PDF list
async function loadPDFList() {
    try {
        const response = await fetch('/api/pdfs');
        const data = await response.json();
        
        if (data.pdfs && data.pdfs.length > 0) {
            pdfList.innerHTML = data.pdfs.map(pdf => `
                <div class="pdf-item">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    ${pdf}
                </div>
            `).join('');
            
            chunkCount.textContent = data.total_chunks || '-';
        } else {
            pdfList.innerHTML = '<div class="loading-pdfs">PDF bulunamadı</div>';
        }
    } catch (error) {
        console.error('PDF listesi yüklenemedi:', error);
        pdfList.innerHTML = '<div class="loading-pdfs">Yükleme hatası</div>';
    }
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isWaitingForResponse) return;
    
    // Hide welcome screen
    if (welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
        messagesContainer.style.display = 'block';
    }
    
    // Add user message
    addMessage(message, 'user');
    messageHistory.push({ role: 'user', content: message });
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0';
    
    // Disable input
    isWaitingForResponse = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Use streaming endpoint for real-time responses
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Server error');
        }
        
        // Remove typing indicator before streaming starts
        removeTypingIndicator(typingId);
        
        // Create streaming message container
        const streamingMsgId = createStreamingMessage();
        
        let fullText = '';
        let metadata = {};
        
        // Process Server-Sent Events (SSE)
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'metadata') {
                            // Update metadata (sources, confidence)
                            metadata = data;
                            updateStreamingMetadata(streamingMsgId, data);
                        } else if (data.type === 'token') {
                            // Append token to text
                            fullText += data.content;
                            updateStreamingContent(streamingMsgId, fullText);
                        } else if (data.type === 'done') {
                            // Finalize message
                            finalizeStreamingMessage(streamingMsgId, fullText, metadata);
                            messageHistory.push({ role: 'bot', content: fullText });
                        } else if (data.type === 'error') {
                            throw new Error(data.content);
                        }
                    } catch (e) {
                        console.error('SSE parse error:', e);
                    }
                }
            }
        }
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage(
            'Bağlantı hatası. Lütfen tekrar deneyin.',
            'bot',
            [],
            0,
            false
        );
        console.error('Stream error:', error);
    } finally {
        isWaitingForResponse = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
}

// Create streaming message container
function createStreamingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    // Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.textContent = '⚖️';
    
    const name = document.createElement('div');
    name.className = 'message-name';
    name.textContent = 'Hukuki AI';
    
    // Streaming badge
    const streamBadge = document.createElement('span');
    streamBadge.className = 'streaming-badge';
    streamBadge.innerHTML = '<span class="pulse"></span> Yazıyor...';
    
    headerDiv.appendChild(avatar);
    headerDiv.appendChild(name);
    headerDiv.appendChild(streamBadge);
    
    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content streaming-content';
    contentDiv.innerHTML = '<span class="cursor-blink">|</span>';
    
    // Metadata container (confidence, sources)
    const metaDiv = document.createElement('div');
    metaDiv.className = 'streaming-metadata';
    metaDiv.style.display = 'none';
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(metaDiv);
    messageDiv.appendChild(contentDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Store references
    messageDiv._contentDiv = contentDiv;
    messageDiv._metaDiv = metaDiv;
    messageDiv._streamBadge = streamBadge;
    
    return messageDiv;
}

// Update streaming metadata (confidence, sources)
function updateStreamingMetadata(container, metadata) {
    const metaDiv = container._metaDiv;
    metaDiv.style.display = 'block';
    
    let html = '';
    
    // Confidence badge
    if (metadata.confidence !== undefined) {
        const confClass = metadata.confidence >= 70 ? 'high' : metadata.confidence >= 50 ? 'medium' : 'low';
        html += `<div class="confidence-badge confidence-${confClass}">
            <span class="confidence-icon">${metadata.confidence >= 70 ? '✅' : metadata.confidence >= 50 ? '⚠️' : '❌'}</span>
            <span class="confidence-text">Güven: ${Math.round(metadata.confidence)}%</span>
        </div>`;
        
        // Low confidence warning
        if (metadata.low_confidence) {
            html += `<div class="low-confidence-warning">
                ⚠️ Düşük güven skoru. Daha spesifik soru deneyebilirsiniz.
            </div>`;
        }
    }
    
    metaDiv.innerHTML = html;
}

// Update streaming content
function updateStreamingContent(container, text) {
    const contentDiv = container._contentDiv;
    // Render markdown for streaming content
    contentDiv.innerHTML = renderMarkdown(text) + '<span class="cursor-blink">|</span>';
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Finalize streaming message
function finalizeStreamingMessage(container, text, metadata) {
    const contentDiv = container._contentDiv;
    const streamBadge = container._streamBadge;
    
    // Final markdown render and remove cursor
    contentDiv.innerHTML = renderMarkdown(text);
    streamBadge.remove();
    
    // Add sources if available
    if (metadata && metadata.sources && metadata.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        let sourcesHTML = '<div class="sources-header">📚 Kaynaklar:</div>';
        metadata.sources.forEach((source, idx) => {
            const scorePercent = Math.round(source.score * 100);
            const scoreClass = scorePercent >= 70 ? 'high' : scorePercent >= 50 ? 'medium' : 'low';
            
            sourcesHTML += `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-name">${source.source}</span>
                        ${source.article ? `<span class="article-badge">${source.article}</span>` : ''}
                        <span class="confidence-badge confidence-${scoreClass}">${scorePercent}%</span>
                    </div>
                </div>
            `;
        });
        
        sourcesDiv.innerHTML = sourcesHTML;
        container.appendChild(sourcesDiv);
    }
    
    // Remove streaming class
    contentDiv.classList.remove('streaming-content');
}

// Add message to chat (non-streaming fallback)
function addMessage(content, role, sources = [], confidence = null, hasSources = false, generated = false, lowConfidence = false, warning = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    // Header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role}-avatar`;
    avatar.textContent = role === 'user' ? '👤' : '⚖️';
    
    const name = document.createElement('div');
    name.className = 'message-name';
    name.textContent = role === 'user' ? 'Siz' : 'Hukuki AI';
    
    // Add LLM badge if generated
    if (role === 'bot' && generated) {
        const llmBadge = document.createElement('span');
        llmBadge.className = 'llm-badge';
        llmBadge.innerHTML = '🦙 Llama-3.2';
        llmBadge.title = 'Llama 3.2-1B-Instruct ile üretildi';
        name.appendChild(llmBadge);
    }
    
    headerDiv.appendChild(avatar);
    headerDiv.appendChild(name);
    
    // Content with markdown rendering
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'bot') {
        // Render markdown for bot messages
        contentDiv.innerHTML = renderMarkdown(content);
    } else {
        // Plain text for user messages
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    
    // Add confidence badge and warning for bot messages
    if (role === 'bot' && confidence !== null) {
        const confDiv = document.createElement('div');
        confDiv.className = 'streaming-metadata';
        
        const confClass = confidence >= 70 ? 'high' : confidence >= 50 ? 'medium' : 'low';
        let confHTML = `<div class="confidence-badge confidence-${confClass}">
            <span class="confidence-icon">${confidence >= 70 ? '✅' : confidence >= 50 ? '⚠️' : '❌'}</span>
            <span class="confidence-text">Güven: ${Math.round(confidence)}%</span>
        </div>`;
        
        if (lowConfidence && warning) {
            confHTML += `<div class="low-confidence-warning">${warning}</div>`;
        }
        
        confDiv.innerHTML = confHTML;
        messageDiv.appendChild(confDiv);
    }
    
    // Add sources for bot messages
    if (role === 'bot' && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        sourcesDiv.innerHTML = `
            <div class="sources-title">
                📚 ${sources.length} Kaynak Bulundu
            </div>
            ${sources.slice(0, 3).map((source, idx) => `
                <div class="source-item">
                    <div class="source-header">
                        <div class="source-name">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                            </svg>
                            ${source.source}
                            ${source.article ? `<span class="article-badge">📜 ${source.article}</span>` : ''}
                        </div>
                        <span class="source-score">${(source.score * 100).toFixed(0)}%</span>
                    </div>
                    <div class="source-preview">${source.preview}</div>
                </div>
            `).join('')}
        `;
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    // Add confidence badge
    if (role === 'bot' && confidence !== null) {
        const badge = document.createElement('div');
        badge.className = 'confidence-badge';
        
        if (confidence >= 0.7) {
            badge.className += ' confidence-high';
            badge.textContent = `✓ Güven: ${(confidence * 100).toFixed(0)}%`;
        } else if (confidence >= 0.5) {
            badge.className += ' confidence-medium';
            badge.textContent = `⚠ Orta Güven: ${(confidence * 100).toFixed(0)}%`;
        } else {
            badge.className += ' confidence-low';
            badge.textContent = `⚠ Düşük Güven: ${(confidence * 100).toFixed(0)}%`;
        }
        
        contentDiv.appendChild(badge);
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    typingDiv.id = id;
    typingDiv.className = 'message';
    
    typingDiv.innerHTML = `
        <div class="message-header">
            <div class="message-avatar bot-avatar">⚖️</div>
            <div class="message-name">Hukuki AI</div>
        </div>
        <div class="typing-indicator">
            <span>Dokümanlarda aranıyor</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return id;
}

function removeTypingIndicator(id) {
    const typing = document.getElementById(id);
    if (typing) typing.remove();
}

// Example questions
function sendExample(button) {
    const text = button.querySelector('.example-text').textContent;
    messageInput.value = text;
    sendMessage();
}

function sendQuickQuestion(question) {
    messageInput.value = question;
    sendMessage();
}

// New chat
function newChat() {
    messagesContainer.innerHTML = '';
    messageHistory = [];
    welcomeScreen.style.display = 'flex';
    messagesContainer.style.display = 'none';
}

// Mobile sidebar toggle
function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('active');
}

