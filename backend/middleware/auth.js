const jwt = require('jsonwebtoken');
const { supabase } = require('../supabase');

/**
 * Comprehensive Authentication Middleware
 * Provides multiple authentication strategies and security features
 */
class AuthMiddleware {
  constructor() {
    this.jwtSecret = process.env.JWT_SECRET;
    this.sessionSecret = process.env.SESSION_SECRET;
  }

  /**
   * Verify JWT token with proper error handling
   */
  async verifyJWT(token) {
    try {
      if (!token) {
        throw new Error('No token provided');
      }

      // Verify with Supabase first (preferred method)
      const { data: { user }, error } = await supabase.auth.getUser(token);
      
      if (error) {
        // Fallback to local JWT verification
        const decoded = jwt.verify(token, this.jwtSecret);
        return { user: decoded, error: null };
      }

      return { user, error: null };
    } catch (error) {
      return { user: null, error: error.message };
    }
  }

  /**
   * Extract token from various sources
   */
  extractToken(req) {
    // Check Authorization header
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      return authHeader.substring(7);
    }

    // Check cookies
    if (req.cookies && req.cookies.token) {
      return req.cookies.token;
    }

    // Check query parameters (for specific endpoints only)
    if (req.query.token && req.path.startsWith('/api/public/')) {
      return req.query.token;
    }

    return null;
  }

  /**
   * Rate limiting configuration
   */
  getRateLimitConfig() {
    return {
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
      max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100, // limit each IP
      message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil((parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000) / 1000)
      },
      standardHeaders: true,
      legacyHeaders: false,
      skip: (req) => {
        // Skip rate limiting for health checks and public endpoints
        return req.path === '/api/health' || req.path.startsWith('/api/public/');
      }
    };
  }

  /**
   * Authentication middleware for protected routes
   */
  authenticate() {
    return async (req, res, next) => {
      try {
        const token = this.extractToken(req);
        
        if (!token) {
          return res.status(401).json({
            success: false,
            error: 'Authentication required',
            code: 'AUTH_REQUIRED'
          });
        }

        const { user, error } = await this.verifyJWT(token);
        
        if (error || !user) {
          return res.status(401).json({
            success: false,
            error: 'Invalid or expired token',
            code: 'INVALID_TOKEN'
          });
        }

        // Add user to request object
        req.user = user;
        req.userId = user.id;
        
        // Add security headers
        res.set({
          'X-User-ID': user.id,
          'X-Authenticated': 'true'
        });

        next();
      } catch (error) {
        console.error('Authentication error:', error);
        return res.status(500).json({
          success: false,
          error: 'Authentication service error',
          code: 'AUTH_ERROR'
        });
      }
    };
  }

  /**
   * Role-based authorization middleware
   */
  authorize(requiredRoles = []) {
    return (req, res, next) => {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required',
          code: 'AUTH_REQUIRED'
        });
      }

      if (requiredRoles.length === 0) {
        return next();
      }

      const userRole = req.user.role || req.user.user_metadata?.role || 'user';
      
      if (!requiredRoles.includes(userRole)) {
        return res.status(403).json({
          success: false,
          error: 'Insufficient permissions',
          code: 'INSUFFICIENT_PERMISSIONS',
          requiredRoles,
          userRole
        });
      }

      next();
    };
  }

  /**
   * Optional authentication middleware
   */
  optionalAuth() {
    return async (req, res, next) => {
      try {
        const token = this.extractToken(req);
        
        if (token) {
          const { user, error } = await this.verifyJWT(token);
          
          if (!error && user) {
            req.user = user;
            req.userId = user.id;
            req.isAuthenticated = true;
          }
        }
        
        next();
      } catch (error) {
        // Continue without authentication
        next();
      }
    };
  }

  /**
   * API key authentication for external services
   */
  apiKeyAuth() {
    return (req, res, next) => {
      const apiKey = req.headers['x-api-key'] || req.headers['authorization'];
      
      if (!apiKey) {
        return res.status(401).json({
          success: false,
          error: 'API key required',
          code: 'API_KEY_REQUIRED'
        });
      }

      // Validate API key (implement your validation logic here)
      const validApiKeys = [
        process.env.FREIGHTOS_API_KEY,
        process.env.DEEPSEEK_API_KEY,
        process.env.MP_API_KEY
      ].filter(Boolean);

      if (!validApiKeys.includes(apiKey.replace('Bearer ', ''))) {
        return res.status(401).json({
          success: false,
          error: 'Invalid API key',
          code: 'INVALID_API_KEY'
        });
      }

      next();
    };
  }

  /**
   * CSRF protection middleware
   */
  csrfProtection() {
    return (req, res, next) => {
      if (req.method === 'GET' || req.method === 'HEAD' || req.method === 'OPTIONS') {
        return next();
      }

      const csrfToken = req.headers['x-csrf-token'] || req.body._csrf;
      const sessionToken = req.session?.csrfToken;

      if (!csrfToken || !sessionToken || csrfToken !== sessionToken) {
        return res.status(403).json({
          success: false,
          error: 'CSRF token validation failed',
          code: 'CSRF_ERROR'
        });
      }

      next();
    };
  }

  /**
   * Input sanitization middleware
   */
  sanitizeInput() {
    return (req, res, next) => {
      // Sanitize request body
      if (req.body) {
        req.body = this.sanitizeObject(req.body);
      }

      // Sanitize query parameters
      if (req.query) {
        req.query = this.sanitizeObject(req.query);
      }

      // Sanitize URL parameters
      if (req.params) {
        req.params = this.sanitizeObject(req.params);
      }

      next();
    };
  }

  /**
   * Sanitize object recursively
   */
  sanitizeObject(obj) {
    if (typeof obj !== 'object' || obj === null) {
      return this.sanitizeValue(obj);
    }

    if (Array.isArray(obj)) {
      return obj.map(item => this.sanitizeObject(item));
    }

    const sanitized = {};
    for (const [key, value] of Object.entries(obj)) {
      sanitized[key] = this.sanitizeObject(value);
    }

    return sanitized;
  }

  /**
   * Sanitize individual values
   */
  sanitizeValue(value) {
    if (typeof value !== 'string') {
      return value;
    }

    // Remove potentially dangerous characters
    return value
      .replace(/[<>]/g, '') // Remove < and >
      .replace(/javascript:/gi, '') // Remove javascript: protocol
      .replace(/on\w+=/gi, '') // Remove event handlers
      .trim();
  }

  /**
   * Security headers middleware
   */
  securityHeaders() {
    return (req, res, next) => {
      // Security headers
      res.set({
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:; frame-ancestors 'none';",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
      });

      next();
    };
  }
}

module.exports = new AuthMiddleware(); 