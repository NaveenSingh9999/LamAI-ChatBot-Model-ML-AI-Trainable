# LamAI Express Server

A modern, ChatGPT-like web interface for the LamAI chatbot built with Express.js, Socket.IO, and vanilla JavaScript.

## Features

### 🚀 Modern Web Interface
- **ChatGPT-like UI**: Clean, responsive design with dark/light theme support
- **Real-time Chat**: WebSocket-based real-time messaging with Socket.IO
- **Session Management**: Persistent chat sessions with history
- **Mobile Responsive**: Optimized for all device sizes

### 🤖 AI Integration
- **Python Backend Integration**: Seamless integration with existing LamAI Python models
- **Mathematical Calculations**: Built-in math expression evaluation
- **Learning Capabilities**: Dynamic knowledge base that learns from conversations
- **Training Interface**: Web-based model training with progress tracking

### 💾 Data Management
- **Chat History**: Persistent storage of conversations
- **Export/Import**: Download chat sessions as JSON
- **Session Restoration**: Resume previous conversations
- **Auto-save**: Automatic conversation backup

### 🔧 Technical Features
- **RESTful API**: Well-structured API endpoints
- **Rate Limiting**: Protection against spam and abuse
- **Error Handling**: Comprehensive error management
- **Security**: Helmet.js security headers and CORS protection
- **Performance**: Compression and optimized asset delivery

## Quick Start

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Install spaCy model (if not already installed)
python -m spacy download en_core_web_lg
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings (optional for development)
nano .env
```

### 3. Start the Server

```bash
# Development mode with auto-reload
npm run dev

# Or production mode
npm start
```

### 4. Access the Application

Open your browser and navigate to:
- **Web Interface**: http://localhost:3000
- **API Health Check**: http://localhost:3000/health
- **API Documentation**: Available in the code comments

## API Endpoints

### Chat API
- `POST /api/chat` - Send a message to the AI
- `GET /api/chat/history` - Get list of chat sessions
- `GET /api/chat/history/:sessionId` - Get specific chat session
- `DELETE /api/chat/history/:sessionId` - Delete a chat session

### Training API
- `POST /api/train` - Start model training

### Health Check
- `GET /health` - Server health status

## API Usage Examples

### Send a Message
```javascript
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: "Hello, how are you?",
        sessionId: "optional-session-id"
    })
});

const data = await response.json();
console.log(data.response); // AI response
```

### Get Chat History
```javascript
const response = await fetch('/api/chat/history');
const sessions = await response.json();
console.log(sessions); // Array of chat sessions
```

### Start Training
```javascript
const response = await fetch('/api/train', {
    method: 'POST'
});
const result = await response.json();
console.log(result.message); // Training status
```

## Socket.IO Events

### Client to Server
- `join-chat` - Join a chat session
- `send-message` - Send a message in real-time

### Server to Client
- `message` - Receive a new message
- `error` - Receive error notifications

## File Structure

```
├── server.js              # Main Express server
├── package.json           # Node.js dependencies
├── .env.example           # Environment variables template
├── public/                # Static web assets
│   ├── index.html        # Main web interface
│   ├── styles.css        # CSS styles
│   └── script.js         # Frontend JavaScript
├── lamai_api.py          # Python AI backend
├── requirements.txt      # Python dependencies
└── training_data/        # Training data files
```

## Configuration

### Environment Variables
- `NODE_ENV`: Development or production mode
- `PORT`: Server port (default: 3000)
- `SESSION_SECRET`: Session encryption key
- `RATE_LIMIT_WINDOW`: Rate limiting window in minutes
- `RATE_LIMIT_MAX`: Maximum requests per window
- `ALLOWED_ORIGINS`: CORS allowed origins

### Customization
- **Themes**: Modify CSS variables in `styles.css`
- **AI Responses**: Edit `lamai_api.py` for custom AI logic
- **Training Data**: Add files to `training_data/` folder
- **UI Components**: Modify `index.html` and `script.js`

## Security Features

- **Rate Limiting**: Prevents spam and DoS attacks
- **Helmet.js**: Security headers for protection
- **CORS**: Cross-origin request protection
- **Input Validation**: Message length and content validation
- **Session Management**: Secure session handling

## Performance Optimization

- **Compression**: Gzip compression for all responses
- **Static Caching**: Efficient static file serving
- **Connection Pooling**: Optimized database connections
- **Asset Minification**: Compressed CSS and JavaScript

## Troubleshooting

### Common Issues

1. **spaCy Model Error**
   ```bash
   python -m spacy download en_core_web_lg
   ```

2. **Port Already in Use**
   ```bash
   # Change PORT in .env file or kill existing process
   lsof -ti:3000 | xargs kill -9
   ```

3. **Python Path Issues**
   ```bash
   # Ensure Python dependencies are installed
   pip install -r requirements.txt
   ```

4. **Socket.IO Connection Issues**
   - Check firewall settings
   - Verify WebSocket support in browser
   - Check CORS configuration

### Debug Mode
Enable detailed logging by setting `NODE_ENV=development` in your `.env` file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation in the code

---

Built with ❤️ using Express.js, Socket.IO, and the LamAI Python backend.
