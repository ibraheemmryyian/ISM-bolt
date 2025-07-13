const prometheus = require('prom-client');
const winston = require('winston');
const { createLogger, format, transports } = winston;
const { combine, timestamp, label, printf, colorize } = format;
const os = require('os');
const axios = require('axios');
const { supabase } = require('../supabase');

// Singleton instance
let instance = null;

class ProductionMonitoring {
  constructor() {
    if (instance) {
      return instance;
    }
    
    this.registry = new prometheus.Registry();
    this.setupMetrics();
    this.setupLogging();
    this.startMetricsCollection();
    
    instance = this;
  }

  /**
   * Setup Prometheus metrics
   */
  setupMetrics() {
    // HTTP request metrics
    this.httpRequestDuration = new prometheus.Histogram({
      name: 'http_request_duration_seconds',
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'route', 'status_code'],
      buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
    });

    this.httpRequestsTotal = new prometheus.Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'route', 'status_code']
    });

    // AI Service metrics
    this.aiProcessingDuration = new prometheus.Histogram({
      name: 'ism_ai_processing_duration_seconds',
      help: 'Duration of AI processing in seconds',
      labelNames: ['service', 'operation'],
      buckets: [1, 5, 10, 30, 60, 120, 300]
    });

    this.aiRequestsTotal = new prometheus.Counter({
      name: 'ism_ai_requests_total',
      help: 'Total number of AI requests',
      labelNames: ['service', 'operation', 'status']
    });

    this.aiErrorsTotal = new prometheus.Counter({
      name: 'ism_ai_errors_total',
      help: 'Total number of AI errors',
      labelNames: ['service', 'operation', 'error_type']
    });

    // Business metrics
    this.matchingSuccessRate = new prometheus.Gauge({
      name: 'ism_matching_success_rate',
      help: 'AI matching success rate (0-1)',
      labelNames: ['algorithm']
    });

    this.symbiosisOpportunities = new prometheus.Gauge({
      name: 'ism_symbiosis_opportunities_count',
      help: 'Number of symbiosis opportunities found',
      labelNames: ['region', 'industry']
    });

    this.carbonReduction = new prometheus.Counter({
      name: 'ism_carbon_reduction_kg',
      help: 'Total carbon reduction in kg CO2e',
      labelNames: ['material_type', 'process']
    });

    // Database metrics
    this.databaseConnections = new prometheus.Gauge({
      name: 'ism_database_connections_active',
      help: 'Number of active database connections'
    });

    this.databaseQueryDuration = new prometheus.Histogram({
      name: 'ism_database_query_duration_seconds',
      help: 'Duration of database queries in seconds',
      labelNames: ['operation'],
      buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
    });

    // External API metrics
    this.externalApiDuration = new prometheus.Histogram({
      name: 'ism_external_api_duration_seconds',
      help: 'Duration of external API calls in seconds',
      labelNames: ['api', 'endpoint'],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30]
    });

    this.externalApiErrors = new prometheus.Counter({
      name: 'ism_external_api_errors_total',
      help: 'Total number of external API errors',
      labelNames: ['api', 'endpoint', 'error_type']
    });

    // System metrics
    this.systemMemoryUsage = new prometheus.Gauge({
      name: 'ism_system_memory_usage_bytes',
      help: 'System memory usage in bytes',
      labelNames: ['type']
    });

    this.systemCpuUsage = new prometheus.Gauge({
      name: 'ism_system_cpu_usage_percent',
      help: 'System CPU usage percentage'
    });

    // User metrics
    this.activeUsers = new prometheus.Gauge({
      name: 'ism_active_users_count',
      help: 'Number of active users',
      labelNames: ['user_type']
    });

    this.userSessions = new prometheus.Counter({
      name: 'ism_user_sessions_total',
      help: 'Total number of user sessions',
      labelNames: ['user_type', 'status']
    });

    // Register all metrics
    this.registry.registerMetric(this.httpRequestDuration);
    this.registry.registerMetric(this.httpRequestsTotal);
    this.registry.registerMetric(this.aiProcessingDuration);
    this.registry.registerMetric(this.aiRequestsTotal);
    this.registry.registerMetric(this.aiErrorsTotal);
    this.registry.registerMetric(this.matchingSuccessRate);
    this.registry.registerMetric(this.symbiosisOpportunities);
    this.registry.registerMetric(this.carbonReduction);
    this.registry.registerMetric(this.databaseConnections);
    this.registry.registerMetric(this.databaseQueryDuration);
    this.registry.registerMetric(this.externalApiDuration);
    this.registry.registerMetric(this.externalApiErrors);
    this.registry.registerMetric(this.systemMemoryUsage);
    this.registry.registerMetric(this.systemCpuUsage);
    this.registry.registerMetric(this.activeUsers);
    this.registry.registerMetric(this.userSessions);
  }

  /**
   * Setup structured logging
   */
  setupLogging() {
    const logFormat = printf(({ timestamp, level, message, service, operation, userId, requestId, ...meta }) => {
      return JSON.stringify({
        timestamp,
        level,
        message,
        service: service || 'ism-backend',
        operation,
        userId,
        requestId,
        ...meta
      });
    });

    this.logger = createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: combine(
        timestamp(),
        label({ label: 'ism-platform' }),
        logFormat
      ),
      defaultMeta: { service: 'ism-backend' },
      transports: [
        new transports.File({ 
          filename: 'logs/error.log', 
          level: 'error',
          maxsize: 5242880, // 5MB
          maxFiles: 5
        }),
        new transports.File({ 
          filename: 'logs/combined.log',
          maxsize: 5242880, // 5MB
          maxFiles: 5
        }),
        new transports.Console({
          format: combine(
            colorize(),
            logFormat
          )
        })
      ]
    });
  }

  /**
   * Start metrics collection
   */
  startMetricsCollection() {
    // Collect system metrics every 30 seconds
    setInterval(() => {
      this.collectSystemMetrics();
    }, 30000);

    // Collect business metrics every 5 minutes
    setInterval(() => {
      this.collectBusinessMetrics();
    }, 300000);
  }

  /**
   * Collect system metrics
   */
  collectSystemMetrics() {
    const memUsage = process.memoryUsage();
    this.systemMemoryUsage.set({ type: 'rss' }, memUsage.rss);
    this.systemMemoryUsage.set({ type: 'heapUsed' }, memUsage.heapUsed);
    this.systemMemoryUsage.set({ type: 'heapTotal' }, memUsage.heapTotal);

    const cpuUsage = process.cpuUsage();
    const totalCpuUsage = (cpuUsage.user + cpuUsage.system) / 1000000; // Convert to seconds
    this.systemCpuUsage.set(totalCpuUsage);
  }

  /**
   * Collect business metrics
   */
  async collectBusinessMetrics() {
    try {
      // Query database for real business metrics
      const { data: matchingData, error: matchingError } = await supabase
        .from('matching_success_rates')
        .select('success_rate')
        .eq('algorithm', 'gnn')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (!matchingError && matchingData) {
        this.matchingSuccessRate.set({ algorithm: 'gnn' }, matchingData.success_rate || 0.7);
      }

      const { data: opportunitiesData, error: opportunitiesError } = await supabase
        .from('symbiosis_opportunities')
        .select('opportunity_count')
        .eq('region', 'global')
        .eq('industry', 'manufacturing')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (!opportunitiesError && opportunitiesData) {
        this.symbiosisOpportunities.set({ region: 'global', industry: 'manufacturing' }, opportunitiesData.opportunity_count || 0);
      }
    } catch (error) {
      this.logger.error('Error collecting business metrics', { error: error.message });
    }
  }

  /**
   * HTTP request middleware
   */
  httpRequestMiddleware() {
    return (req, res, next) => {
      const start = Date.now();
      const originalSend = res.send;

      res.send = function(data) {
        const duration = (Date.now() - start) / 1000;
        const route = req.route ? req.route.path : req.path;
        
        this.httpRequestDuration.observe(
          { method: req.method, route, status_code: res.statusCode },
          duration
        );
        
        this.httpRequestsTotal.inc({
          method: req.method,
          route,
          status_code: res.statusCode
        });

        originalSend.call(this, data);
      }.bind(this);

      next();
    };
  }

  /**
   * AI service monitoring
   */
  trackAIRequest(service, operation) {
    const start = Date.now();
    
    return {
      success: () => {
        const duration = (Date.now() - start) / 1000;
        this.aiProcessingDuration.observe({ service, operation }, duration);
        this.aiRequestsTotal.inc({ service, operation, status: 'success' });
      },
      error: (errorType) => {
        const duration = (Date.now() - start) / 1000;
        this.aiProcessingDuration.observe({ service, operation }, duration);
        this.aiRequestsTotal.inc({ service, operation, status: 'error' });
        this.aiErrorsTotal.inc({ service, operation, error_type: errorType });
      }
    };
  }

  /**
   * Database monitoring
   */
  trackDatabaseQuery(operation) {
    const start = Date.now();
    
    return {
      success: () => {
        const duration = (Date.now() - start) / 1000;
        this.databaseQueryDuration.observe({ operation }, duration);
      },
      error: () => {
        const duration = (Date.now() - start) / 1000;
        this.databaseQueryDuration.observe({ operation }, duration);
      }
    };
  }

  /**
   * External API monitoring
   */
  trackExternalAPI(api, endpoint) {
    const start = Date.now();
    
    return {
      success: () => {
        const duration = (Date.now() - start) / 1000;
        this.externalApiDuration.observe({ api, endpoint }, duration);
      },
      error: (errorType) => {
        const duration = (Date.now() - start) / 1000;
        this.externalApiDuration.observe({ api, endpoint }, duration);
        this.externalApiErrors.inc({ api, endpoint, error_type: errorType });
      }
    };
  }

  /**
   * Business metrics tracking
   */
  trackMatchingSuccess(algorithm, successRate) {
    this.matchingSuccessRate.set({ algorithm }, successRate);
  }

  trackSymbiosisOpportunity(region, industry, count) {
    this.symbiosisOpportunities.set({ region, industry }, count);
  }

  trackCarbonReduction(materialType, process, kgCO2e) {
    this.carbonReduction.inc({ material_type: materialType, process }, kgCO2e);
  }

  /**
   * User metrics tracking
   */
  trackUserSession(userType, status) {
    this.userSessions.inc({ user_type: userType, status });
  }

  setActiveUsers(userType, count) {
    this.activeUsers.set({ user_type: userType }, count);
  }

  /**
   * Get metrics for Prometheus
   */
  async getMetrics() {
    return await this.registry.metrics();
  }

  /**
   * Health check endpoint
   */
  async healthCheck() {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      version: process.env.npm_package_version || '1.0.0'
    };

    // Check critical services
    try {
      // Check database connection
      // await this.checkDatabaseHealth();
      
      // Check AI services
      await this.checkAIServicesHealth();
      
      return health;
    } catch (error) {
      health.status = 'unhealthy';
      health.error = error.message;
      return health;
    }
  }

  /**
   * Check AI services health
   */
  async checkAIServicesHealth() {
    const aiServices = [
      { name: 'gnn-reasoning', url: 'http://localhost:5001/api/ai/gnn/models' },
      { name: 'revolutionary-matching', url: 'http://localhost:5001/api/ai/services/status' },
      { name: 'knowledge-graph', url: 'http://localhost:5001/api/ai/knowledge-graph/build' },
      { name: 'adaptive-onboarding', url: 'http://localhost:5001/api/adaptive-onboarding/start' }
    ];

    const healthChecks = await Promise.allSettled(
      aiServices.map(async (service) => {
        try {
          const response = await axios.get(service.url, { timeout: 5000 });
          return { name: service.name, status: response.status === 200 ? 'healthy' : 'unhealthy' };
        } catch (error) {
          // For POST endpoints, try a health check instead
          if (service.url.includes('/start') || service.url.includes('/build')) {
            try {
              const healthResponse = await axios.get('http://localhost:5001/api/health', { timeout: 5000 });
              return { name: service.name, status: healthResponse.status === 200 ? 'healthy' : 'unhealthy' };
            } catch (healthError) {
              return { name: service.name, status: 'unhealthy', error: healthError.message };
            }
          }
          return { name: service.name, status: 'unhealthy', error: error.message };
        }
      })
    );

    return healthChecks.map((result, index) => ({
      service: aiServices[index].name,
      status: result.status === 'fulfilled' ? result.value.status : 'unhealthy',
      error: result.status === 'rejected' ? result.reason.message : null
    }));
  }

  /**
   * Structured logging methods
   */
  log(level, message, meta = {}) {
    this.logger.log(level, message, meta);
  }

  info(message, meta = {}) {
    this.logger.info(message, meta);
  }

  warn(message, meta = {}) {
    this.logger.warn(message, meta);
  }

  error(message, meta = {}) {
    this.logger.error(message, meta);
  }

  debug(message, meta = {}) {
    this.logger.debug(message, meta);
  }

  /**
   * Log AI operation
   */
  logAIOperation(operation, service, userId, requestId, details = {}) {
    this.info(`AI Operation: ${operation}`, {
      service,
      userId,
      requestId,
      operation,
      ...details
    });
  }

  /**
   * Log business event
   */
  logBusinessEvent(event, userId, requestId, details = {}) {
    this.info(`Business Event: ${event}`, {
      userId,
      requestId,
      event,
      ...details
    });
  }

  /**
   * Log security event
   */
  logSecurityEvent(event, userId, ip, details = {}) {
    this.warn(`Security Event: ${event}`, {
      userId,
      ip,
      event,
      ...details
    });
  }

  /**
   * Get singleton instance
   */
  static getInstance() {
    if (!instance) {
      instance = new ProductionMonitoring();
    }
    return instance;
  }
}

module.exports = ProductionMonitoring; 