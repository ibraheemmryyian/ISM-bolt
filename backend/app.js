const express = require('express');
const { PythonShell } = require('python-shell');
const crypto = require('crypto');
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(express.json());

// Test route
app.get('/api', (req, res) => {
  res.json({ message: 'Industrial AI Marketplace API' });
});

// AI matching endpoint
app.post('/api/match', (req, res) => {
  const { buyer, seller } = req.body;
  
  // Run revolutionary AI matching
  const options = {
    mode: 'text',
    pythonOptions: ['-u'], // unbuffered output
    args: [JSON.stringify({ buyer, seller })]
  };

  PythonShell.run('revolutionary_ai_matching.py', options)
    .then(result => {
      try {
        const matchResult = JSON.parse(result[0]);
        
        // Simulate blockchain logging
        const transactionData = {
          buyerId: buyer.id,
          sellerId: seller.id,
          timestamp: new Date().toISOString(),
          score: matchResult.revolutionary_score
        };
        const txHash = crypto.createHash('sha256')
          .update(JSON.stringify(transactionData))
          .digest('hex');
        
        res.json({
          ...matchResult,
          transactionHash: txHash,
          blockchainStatus: "preparing full blockchain ledger"
        });
      } catch (e) {
        res.status(500).json({ error: 'Failed to parse AI result' });
      }
    })
    .catch(err => {
      res.status(500).json({ error: 'AI matching failed', details: err.message });
    });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
