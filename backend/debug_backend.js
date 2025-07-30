// Debug backend to troubleshoot startup issues
require('dotenv').config();

console.log('🔧 Debug Backend Starting...');
console.log('Current working directory:', process.cwd());
console.log('PORT env variable:', process.env.PORT);
console.log('All env PORT variables:', Object.keys(process.env).filter(k => k.includes('PORT')));

const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000; // Hardcode to avoid any env variable issues

console.log('Setting up middleware...');
app.use(cors());
app.use(express.json());

console.log('Setting up routes...');

app.get('/api/test', (req, res) => {
  console.log('Test endpoint hit!');
  res.json({
    status: 'OK',
    message: 'Debug backend is working!',
    timestamp: new Date().toISOString(),
    port: PORT
  });
});

app.get('/api/health', (req, res) => {
  console.log('Health endpoint hit!');
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    port: PORT
  });
});

console.log(`Attempting to start server on port ${PORT}...`);

try {
  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Debug Backend Server running on port ${PORT}`);
    console.log(`📡 Test at: http://localhost:${PORT}/api/test`);
    console.log(`🏥 Health at: http://localhost:${PORT}/api/health`);
    console.log('Server is ready!');
  });

  server.on('error', (err) => {
    console.error('❌ Server error:', err);
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${PORT} is already in use!`);
    }
  });
} catch (error) {
  console.error('❌ Failed to start server:', error);
}