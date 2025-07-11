const request = require('supertest');
const app = require('../app');

describe('Backend API Tests', () => {
  describe('Health Check', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'healthy');
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

  describe('Error Handling', () => {
    it('should handle 404 for unknown routes', async () => {
      await request(app)
        .get('/api/unknown-route')
        .expect(404);
    });

    it('should handle malformed JSON', async () => {
      await request(app)
        .post('/api/ai-infer-listings')
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);
    });
  });
}); 