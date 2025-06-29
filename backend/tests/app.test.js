const request = require('supertest');
const app = require('../app');

describe('Backend API Tests', () => {
  describe('Health Check', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('version');
    });
  });

  describe('Base API', () => {
    it('should return API message', async () => {
      const response = await request(app)
        .get('/api')
        .expect(200);

      expect(response.body).toHaveProperty('message', 'Industrial AI Marketplace API');
    });
  });

  describe('AI Inference Endpoint', () => {
    const validFormData = {
      companyName: 'Test Company',
      industry: 'Manufacturing',
      products: 'Steel products',
      location: 'New York',
      productionVolume: '1000 tons/year',
      mainMaterials: 'Iron ore, coal',
      processDescription: 'Steel manufacturing process'
    };

    it('should validate required fields', async () => {
      const response = await request(app)
        .post('/api/ai-infer-listings')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
      expect(response.body).toHaveProperty('details');
    });

    it('should validate field lengths', async () => {
      const invalidData = {
        ...validFormData,
        companyName: 'a'.repeat(101) // Too long
      };

      const response = await request(app)
        .post('/api/ai-infer-listings')
        .send(invalidData)
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid form data', async () => {
      const response = await request(app)
        .post('/api/ai-infer-listings')
        .send(validFormData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('listings');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('AI Matching Endpoint', () => {
    const validMatchData = {
      buyer: {
        id: 'buyer123',
        industry: 'Automotive',
        needs: ['steel', 'aluminum']
      },
      seller: {
        id: 'seller456',
        industry: 'Metallurgy',
        products: ['steel', 'aluminum']
      }
    };

    it('should validate buyer and seller objects', async () => {
      const response = await request(app)
        .post('/api/match')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate buyer and seller IDs', async () => {
      const invalidData = {
        buyer: { industry: 'Automotive' },
        seller: { industry: 'Metallurgy' }
      };

      const response = await request(app)
        .post('/api/match')
        .send(invalidData)
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid match data', async () => {
      const response = await request(app)
        .post('/api/match')
        .send(validMatchData)
        .expect(200);

      expect(response.body).toHaveProperty('transactionHash');
      expect(response.body).toHaveProperty('blockchainStatus');
    });
  });

  describe('User Feedback Endpoint', () => {
    const validFeedback = {
      matchId: 'match123',
      userId: 'user456',
      rating: 5,
      feedback: 'Great match!',
      categories: ['quality', 'delivery']
    };

    it('should validate required feedback fields', async () => {
      const response = await request(app)
        .post('/api/feedback')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate rating range', async () => {
      const invalidFeedback = {
        ...validFeedback,
        rating: 6 // Invalid rating
      };

      const response = await request(app)
        .post('/api/feedback')
        .send(invalidFeedback)
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid feedback', async () => {
      const response = await request(app)
        .post('/api/feedback')
        .send(validFeedback)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('message');
    });
  });

  describe('Symbiosis Network Endpoint', () => {
    const validParticipants = [
      {
        id: 'company1',
        industry: 'Steel Manufacturing',
        annual_waste: 1000,
        carbon_footprint: 500,
        waste_type: 'Slag',
        location: 'Pittsburgh'
      },
      {
        id: 'company2',
        industry: 'Cement Production',
        annual_waste: 500,
        carbon_footprint: 300,
        waste_type: 'Fly ash',
        location: 'Chicago'
      }
    ];

    it('should validate participants array', async () => {
      const response = await request(app)
        .post('/api/symbiosis-network')
        .send({ participants: [] })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate participant structure', async () => {
      const invalidParticipants = [
        { industry: 'Steel' } // Missing required id
      ];

      const response = await request(app)
        .post('/api/symbiosis-network')
        .send({ participants: invalidParticipants })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid participants', async () => {
      const response = await request(app)
        .post('/api/symbiosis-network')
        .send({ participants: validParticipants })
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('network');
    });
  });

  describe('Explainable AI Endpoint', () => {
    const validData = {
      buyer: {
        id: 'buyer123',
        industry: 'Automotive'
      },
      seller: {
        id: 'seller456',
        industry: 'Metallurgy'
      }
    };

    it('should validate buyer and seller objects', async () => {
      const response = await request(app)
        .post('/api/explain-match')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid explanation request', async () => {
      const response = await request(app)
        .post('/api/explain-match')
        .send(validData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('explanation');
    });
  });

  describe('Save Listings Endpoint', () => {
    const validData = {
      listings: [
        {
          id: 'listing1',
          title: 'Steel Products',
          description: 'High-quality steel'
        }
      ],
      userId: 'user123'
    };

    it('should validate listings array', async () => {
      const response = await request(app)
        .post('/api/save-listings')
        .send({ listings: [], userId: 'user123' })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate userId', async () => {
      const response = await request(app)
        .post('/api/save-listings')
        .send({ listings: validData.listings })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid listings data', async () => {
      const response = await request(app)
        .post('/api/save-listings')
        .send(validData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('message');
    });
  });

  describe('AI Chat Endpoint', () => {
    const validData = {
      message: 'Hello, I need help with steel procurement',
      context: { userId: 'user123' }
    };

    it('should validate message', async () => {
      const response = await request(app)
        .post('/api/ai-chat')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate message length', async () => {
      const invalidData = {
        message: 'a'.repeat(1001), // Too long
        context: {}
      };

      const response = await request(app)
        .post('/api/ai-chat')
        .send(invalidData)
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid chat message', async () => {
      const response = await request(app)
        .post('/api/ai-chat')
        .send(validData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('response');
    });
  });

  describe('Real-time Recommendations Endpoint', () => {
    const validData = {
      userId: 'user123',
      userProfile: {
        industry: 'Automotive',
        preferences: ['sustainability', 'cost-effectiveness']
      }
    };

    it('should validate userId', async () => {
      const response = await request(app)
        .post('/api/real-time-recommendations')
        .send({ userProfile: {} })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should validate userProfile', async () => {
      const response = await request(app)
        .post('/api/real-time-recommendations')
        .send({ userId: 'user123' })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should accept valid recommendation request', async () => {
      const response = await request(app)
        .post('/api/real-time-recommendations')
        .send(validData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('recommendations');
    });
  });

  describe('Error Handling', () => {
    it('should handle 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/api/unknown-route')
        .expect(404);
    });

    it('should handle malformed JSON', async () => {
      const response = await request(app)
        .post('/api/ai-infer-listings')
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limits', async () => {
      // Make multiple requests quickly
      const promises = Array(105).fill().map(() => 
        request(app).get('/api/health')
      );

      const responses = await Promise.all(promises);
      const rateLimited = responses.some(res => res.status === 429);

      expect(rateLimited).toBe(true);
    });
  });
}); 