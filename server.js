const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: process.env.NODE_ENV === 'production' ? false : ['http://localhost:3000', 'http://127.0.0.1:3000'],
    methods: ['GET', 'POST']
  }
});

const PORT = process.env.PORT || 3000;
const PYTHON_SCRIPT_PATH = path.join(__dirname, 'lamai_api.py');
const CHAT_HISTORY_FILE = path.join(__dirname, 'chat_history.json');

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  }
}));
app.use(compression());
app.use(morgan('combined'));
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(limiter);

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Chat session storage
const chatSessions = new Map();

// Utility functions
const loadChatHistory = async () => {
  try {
    const data = await fs.readFile(CHAT_HISTORY_FILE, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    return [];
  }
};

const saveChatHistory = async (history) => {
  try {
    await fs.writeFile(CHAT_HISTORY_FILE, JSON.stringify(history, null, 2));
  } catch (error) {
    console.error('Error saving chat history:', error);
  }
};

const callPythonAPI = (query) => {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', ['-c', `
import sys
sys.path.append('${__dirname}')
from lamai_api import process_query
query = "${query.replace(/"/g, '\\"')}"
result = process_query(query)
print(result)
    `]);

    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        resolve(output.trim());
      } else {
        reject(new Error(`Python script failed: ${error}`));
      }
    });

    setTimeout(() => {
      python.kill();
      reject(new Error('Python script timeout'));
    }, 30000); // 30 second timeout
  });
};

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: require('./package.json').version
  });
});

// Chat API endpoints
app.post('/api/chat', async (req, res) => {
  try {
    const { message, sessionId = uuidv4() } = req.body;

    if (!message || typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({
        error: 'Message is required and must be a non-empty string'
      });
    }

    // Get or create chat session
    if (!chatSessions.has(sessionId)) {
      chatSessions.set(sessionId, {
        id: sessionId,
        messages: [],
        createdAt: new Date().toISOString()
      });
    }

    const session = chatSessions.get(sessionId);
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString()
    };

    session.messages.push(userMessage);

    // Call Python API
    const aiResponse = await callPythonAPI(message.trim());
    
    const assistantMessage = {
      id: uuidv4(),
      role: 'assistant',
      content: aiResponse,
      timestamp: new Date().toISOString()
    };

    session.messages.push(assistantMessage);

    // Save to persistent storage
    const history = await loadChatHistory();
    const existingSessionIndex = history.findIndex(s => s.id === sessionId);
    
    if (existingSessionIndex >= 0) {
      history[existingSessionIndex] = session;
    } else {
      history.push(session);
    }
    
    await saveChatHistory(history);

    res.json({
      response: aiResponse,
      sessionId: sessionId,
      messageId: assistantMessage.id,
      timestamp: assistantMessage.timestamp
    });

  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

app.get('/api/chat/history/:sessionId?', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const history = await loadChatHistory();

    if (sessionId) {
      const session = history.find(s => s.id === sessionId);
      if (!session) {
        return res.status(404).json({ error: 'Session not found' });
      }
      res.json(session);
    } else {
      // Return list of sessions with metadata
      const sessions = history.map(session => ({
        id: session.id,
        createdAt: session.createdAt,
        messageCount: session.messages.length,
        lastMessage: session.messages[session.messages.length - 1]?.content.substring(0, 100) + '...'
      }));
      res.json(sessions);
    }
  } catch (error) {
    console.error('History API error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

app.delete('/api/chat/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const history = await loadChatHistory();
    const filteredHistory = history.filter(s => s.id !== sessionId);
    
    await saveChatHistory(filteredHistory);
    chatSessions.delete(sessionId);
    
    res.json({ message: 'Session deleted successfully' });
  } catch (error) {
    console.error('Delete session error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Training API
app.post('/api/train', async (req, res) => {
  try {
    const python = spawn('python3', ['-c', `
import sys
sys.path.append('${__dirname}')
from lamai_api import train_from_files
train_from_files()
print("Training completed successfully")
    `]);

    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        res.json({
          message: 'Training completed successfully',
          output: output.trim()
        });
      } else {
        res.status(500).json({
          error: 'Training failed',
          details: error
        });
      }
    });

  } catch (error) {
    console.error('Training API error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Socket.IO for real-time chat
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('join-chat', (sessionId) => {
    socket.join(sessionId);
    console.log(`User ${socket.id} joined chat session: ${sessionId}`);
  });

  socket.on('send-message', async (data) => {
    try {
      const { message, sessionId } = data;
      
      if (!message || !sessionId) {
        socket.emit('error', { message: 'Message and sessionId are required' });
        return;
      }

      // Broadcast user message to room
      socket.to(sessionId).emit('message', {
        id: uuidv4(),
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
      });

      // Get AI response
      const aiResponse = await callPythonAPI(message);
      
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: aiResponse,
        timestamp: new Date().toISOString()
      };

      // Broadcast AI response to room
      io.to(sessionId).emit('message', assistantMessage);

    } catch (error) {
      console.error('Socket message error:', error);
      socket.emit('error', { message: 'Failed to process message' });
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Express error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: 'The requested resource was not found'
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

server.listen(PORT, () => {
  console.log(`🚀 LamAI Express Server is running on port ${PORT}`);
  console.log(`📱 Web interface: http://localhost:${PORT}`);
  console.log(`🔗 API endpoint: http://localhost:${PORT}/api/chat`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
