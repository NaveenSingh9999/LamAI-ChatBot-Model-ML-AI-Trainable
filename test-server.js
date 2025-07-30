const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Basic middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Test route
app.get('/test', (req, res) => {
    res.json({ message: 'Server is working!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Test server running on http://localhost:${PORT}`);
    console.log(`📱 Test endpoint: http://localhost:${PORT}/test`);
});
