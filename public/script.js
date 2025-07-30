// Global variables
let socket;
let currentSessionId = null;
let messageHistory = [];
let isConnected = false;
let isTyping = false;

// DOM elements
const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    newChatBtn: document.getElementById('newChatBtn'),
    chatHistory: document.getElementById('chatHistory'),
    trainBtn: document.getElementById('trainBtn'),
    settingsBtn: document.getElementById('settingsBtn'),
    chatTitle: document.getElementById('chatTitle'),
    statusIndicator: document.getElementById('statusIndicator'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    exportChatBtn: document.getElementById('exportChatBtn'),
    chatContainer: document.getElementById('chatContainer'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    messages: document.getElementById('messages'),
    typingIndicator: document.getElementById('typingIndicator'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    charCounter: document.getElementById('charCounter'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettingsModal: document.getElementById('closeSettingsModal'),
    trainingModal: document.getElementById('trainingModal'),
    closeTrainingModal: document.getElementById('closeTrainingModal'),
    startTrainingBtn: document.getElementById('startTrainingBtn'),
    trainingStatus: document.getElementById('trainingStatus'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    toastContainer: document.getElementById('toastContainer')
};

// Utility functions
const generateId = () => Math.random().toString(36).substr(2, 9);

const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    
    return date.toLocaleDateString();
};

const showToast = (message, type = 'success') => {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
};

const showLoading = (show = true) => {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
};

const updateConnectionStatus = (connected) => {
    isConnected = connected;
    const statusIcon = elements.statusIndicator.querySelector('i');
    const statusText = elements.statusIndicator.querySelector('span');
    
    if (connected) {
        statusIcon.className = 'fas fa-circle';
        statusText.textContent = 'Online';
        elements.statusIndicator.style.color = 'var(--success-color)';
    } else {
        statusIcon.className = 'fas fa-circle';
        statusText.textContent = 'Offline';
        elements.statusIndicator.style.color = 'var(--error-color)';
    }
};

// Socket.IO connection
const initializeSocket = () => {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
        
        if (currentSessionId) {
            socket.emit('join-chat', currentSessionId);
        }
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });
    
    socket.on('message', (message) => {
        addMessage(message.content, message.role, message.timestamp);
        hideTyping();
    });
    
    socket.on('error', (error) => {
        console.error('Socket error:', error);
        showToast(error.message || 'Connection error', 'error');
        hideTyping();
    });
};

// Message handling
const addMessage = (content, role = 'user', timestamp = new Date().toISOString()) => {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">
                ${content.replace(/\n/g, '<br>')}
            </div>
            <div class="message-time">${formatTime(timestamp)}</div>
        </div>
    `;
    
    elements.messages.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    
    // Hide welcome message
    if (elements.welcomeMessage) {
        elements.welcomeMessage.style.display = 'none';
    }
    
    messageHistory.push({ content, role, timestamp });
};

const showTyping = () => {
    if (!isTyping) {
        isTyping = true;
        elements.typingIndicator.style.display = 'flex';
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }
};

const hideTyping = () => {
    isTyping = false;
    elements.typingIndicator.style.display = 'none';
};

const sendMessage = async () => {
    const message = elements.messageInput.value.trim();
    
    if (!message || !isConnected) {
        return;
    }
    
    // Disable input
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    elements.messageInput.value = '';
    updateCharCounter();
    
    // Show typing indicator
    showTyping();
    
    try {
        if (!currentSessionId) {
            currentSessionId = generateId();
            socket.emit('join-chat', currentSessionId);
            updateChatTitle();
        }
        
        // Send via REST API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                sessionId: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addMessage(data.response, 'assistant', data.timestamp);
        } else {
            throw new Error(data.error || 'Failed to send message');
        }
        
    } catch (error) {
        console.error('Send message error:', error);
        showToast('Failed to send message. Please try again.', 'error');
        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    } finally {
        hideTyping();
        elements.messageInput.disabled = false;
        elements.sendBtn.disabled = false;
        elements.messageInput.focus();
    }
};

const updateCharCounter = () => {
    const length = elements.messageInput.value.length;
    const maxLength = elements.messageInput.getAttribute('maxlength') || 4000;
    elements.charCounter.textContent = `${length} / ${maxLength}`;
    
    // Update send button state
    elements.sendBtn.disabled = length === 0 || !isConnected;
};

const updateChatTitle = () => {
    if (currentSessionId && messageHistory.length > 0) {
        const firstMessage = messageHistory.find(m => m.role === 'user');
        if (firstMessage) {
            const title = firstMessage.content.substring(0, 30) + (firstMessage.content.length > 30 ? '...' : '');
            elements.chatTitle.textContent = title;
        }
    } else {
        elements.chatTitle.textContent = 'LamAI Assistant';
    }
};

// Chat history management
const loadChatHistory = async () => {
    try {
        const response = await fetch('/api/chat/history');
        const sessions = await response.json();
        
        elements.chatHistory.innerHTML = '';
        
        if (sessions.length === 0) {
            elements.chatHistory.innerHTML = `
                <div class="history-empty">
                    <i class="fas fa-comments"></i>
                    <span>No chat history yet</span>
                </div>
            `;
            return;
        }
        
        sessions.forEach(session => {
            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'chat-session';
            sessionDiv.dataset.sessionId = session.id;
            
            sessionDiv.innerHTML = `
                <div class="session-title">Chat Session</div>
                <div class="session-preview">${session.lastMessage}</div>
                <div class="session-date">${formatTime(session.createdAt)}</div>
            `;
            
            sessionDiv.addEventListener('click', () => loadChatSession(session.id));
            elements.chatHistory.appendChild(sessionDiv);
        });
        
    } catch (error) {
        console.error('Load history error:', error);
        elements.chatHistory.innerHTML = `
            <div class="history-error">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Failed to load history</span>
            </div>
        `;
    }
};

const loadChatSession = async (sessionId) => {
    try {
        showLoading(true);
        
        const response = await fetch(`/api/chat/history/${sessionId}`);
        const session = await response.json();
        
        if (response.ok) {
            currentSessionId = sessionId;
            messageHistory = session.messages || [];
            
            // Clear current messages
            elements.messages.innerHTML = '';
            elements.welcomeMessage.style.display = 'none';
            
            // Load messages
            messageHistory.forEach(msg => {
                addMessage(msg.content, msg.role, msg.timestamp);
            });
            
            // Update UI
            updateChatTitle();
            socket.emit('join-chat', sessionId);
            
            // Update active session in sidebar
            document.querySelectorAll('.chat-session').forEach(el => {
                el.classList.remove('active');
            });
            document.querySelector(`[data-session-id="${sessionId}"]`)?.classList.add('active');
            
        } else {
            throw new Error(session.error || 'Failed to load session');
        }
        
    } catch (error) {
        console.error('Load session error:', error);
        showToast('Failed to load chat session', 'error');
    } finally {
        showLoading(false);
    }
};

const startNewChat = () => {
    currentSessionId = null;
    messageHistory = [];
    
    // Clear messages
    elements.messages.innerHTML = '';
    elements.welcomeMessage.style.display = 'flex';
    
    // Reset title
    elements.chatTitle.textContent = 'LamAI Assistant';
    
    // Remove active session
    document.querySelectorAll('.chat-session').forEach(el => {
        el.classList.remove('active');
    });
    
    // Focus input
    elements.messageInput.focus();
};

const clearCurrentChat = async () => {
    if (!currentSessionId) {
        startNewChat();
        return;
    }
    
    if (confirm('Are you sure you want to clear this chat? This action cannot be undone.')) {
        try {
            const response = await fetch(`/api/chat/history/${currentSessionId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                startNewChat();
                loadChatHistory();
                showToast('Chat cleared successfully');
            } else {
                throw new Error('Failed to clear chat');
            }
            
        } catch (error) {
            console.error('Clear chat error:', error);
            showToast('Failed to clear chat', 'error');
        }
    }
};

const exportChat = () => {
    if (messageHistory.length === 0) {
        showToast('No messages to export', 'warning');
        return;
    }
    
    const chatData = {
        sessionId: currentSessionId,
        exportDate: new Date().toISOString(),
        messages: messageHistory
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lamai-chat-${currentSessionId || 'export'}-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showToast('Chat exported successfully');
};

// Training functionality
const startTraining = async () => {
    try {
        showLoading(true);
        elements.startTrainingBtn.disabled = true;
        elements.startTrainingBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
        
        elements.trainingStatus.innerHTML = `
            <div class="status-training">
                <i class="fas fa-spinner fa-spin"></i>
                <span>Training in progress...</span>
            </div>
        `;
        
        const response = await fetch('/api/train', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            elements.trainingStatus.innerHTML = `
                <div class="status-success">
                    <i class="fas fa-check-circle"></i>
                    <span>Training completed successfully</span>
                </div>
            `;
            showToast('Model training completed successfully');
        } else {
            throw new Error(data.error || 'Training failed');
        }
        
    } catch (error) {
        console.error('Training error:', error);
        elements.trainingStatus.innerHTML = `
            <div class="status-error">
                <i class="fas fa-exclamation-circle"></i>
                <span>Training failed: ${error.message}</span>
            </div>
        `;
        showToast('Training failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
        elements.startTrainingBtn.disabled = false;
        elements.startTrainingBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
    }
};

// Settings management
const loadSettings = () => {
    const theme = localStorage.getItem('theme') || 'light';
    const fontSize = localStorage.getItem('fontSize') || '14';
    const soundEffects = localStorage.getItem('soundEffects') !== 'false';
    const autoSave = localStorage.getItem('autoSave') !== 'false';
    
    document.getElementById('themeSelect').value = theme;
    document.getElementById('fontSizeRange').value = fontSize;
    document.getElementById('fontSizeValue').textContent = fontSize + 'px';
    document.getElementById('soundEffects').checked = soundEffects;
    document.getElementById('autoSave').checked = autoSave;
    
    applyTheme(theme);
    applyFontSize(fontSize);
};

const saveSettings = () => {
    const theme = document.getElementById('themeSelect').value;
    const fontSize = document.getElementById('fontSizeRange').value;
    const soundEffects = document.getElementById('soundEffects').checked;
    const autoSave = document.getElementById('autoSave').checked;
    
    localStorage.setItem('theme', theme);
    localStorage.setItem('fontSize', fontSize);
    localStorage.setItem('soundEffects', soundEffects);
    localStorage.setItem('autoSave', autoSave);
    
    applyTheme(theme);
    applyFontSize(fontSize);
    
    showToast('Settings saved successfully');
};

const applyTheme = (theme) => {
    if (theme === 'auto') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    } else {
        document.documentElement.setAttribute('data-theme', theme);
    }
};

const applyFontSize = (fontSize) => {
    document.documentElement.style.setProperty('--font-size', fontSize + 'px');
};

// Event listeners
const setupEventListeners = () => {
    // Sidebar toggle
    elements.sidebarToggle.addEventListener('click', () => {
        elements.sidebar.classList.toggle('open');
    });
    
    // New chat
    elements.newChatBtn.addEventListener('click', startNewChat);
    
    // Clear chat
    elements.clearChatBtn.addEventListener('click', clearCurrentChat);
    
    // Export chat
    elements.exportChatBtn.addEventListener('click', exportChat);
    
    // Message input
    elements.messageInput.addEventListener('input', updateCharCounter);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send button
    elements.sendBtn.addEventListener('click', sendMessage);
    
    // Training
    elements.trainBtn.addEventListener('click', () => {
        elements.trainingModal.classList.add('show');
    });
    elements.startTrainingBtn.addEventListener('click', startTraining);
    elements.closeTrainingModal.addEventListener('click', () => {
        elements.trainingModal.classList.remove('show');
    });
    
    // Settings
    elements.settingsBtn.addEventListener('click', () => {
        elements.settingsModal.classList.add('show');
    });
    elements.closeSettingsModal.addEventListener('click', () => {
        elements.settingsModal.classList.remove('show');
        saveSettings();
    });
    
    // Settings controls
    document.getElementById('fontSizeRange').addEventListener('input', (e) => {
        document.getElementById('fontSizeValue').textContent = e.target.value + 'px';
    });
    
    // Modal close on outside click
    [elements.settingsModal, elements.trainingModal].forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('show');
                if (modal === elements.settingsModal) {
                    saveSettings();
                }
            }
        });
    });
    
    // Auto-resize textarea
    elements.messageInput.addEventListener('input', () => {
        elements.messageInput.style.height = 'auto';
        elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
    });
    
    // Theme change detection
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        const theme = localStorage.getItem('theme') || 'light';
        if (theme === 'auto') {
            applyTheme(theme);
        }
    });
};

// Initialize application
const init = () => {
    console.log('Initializing LamAI Chat Application');
    
    // Load settings
    loadSettings();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize socket connection
    initializeSocket();
    
    // Load chat history
    loadChatHistory();
    
    // Update character counter
    updateCharCounter();
    
    // Focus message input
    elements.messageInput.focus();
    
    console.log('LamAI Chat Application initialized successfully');
};

// Start the application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
