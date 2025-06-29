const Sentry = require('@sentry/node');

// Initialize Sentry if DSN is provided
if (process.env.SENTRY_DSN) {
  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    release: process.env.npm_package_version || '1.0.0',
    
    // Performance monitoring
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    
    // Error filtering
    beforeSend(event, hint) {
      // Filter out certain errors
      if (hint.originalException && hint.originalException.code === 'ECONNRESET') {
        return null;
      }
      return event;
    },
    
    // Integrations
    integrations: [
      new Sentry.Integrations.Http({ tracing: true }),
      new Sentry.Integrations.Express({ app: require('./app') }),
    ],
    
    // Debug mode in development
    debug: process.env.NODE_ENV === 'development',
  });
  
  console.log('✅ Sentry initialized for error monitoring');
} else {
  console.log('⚠️  Sentry DSN not provided, error monitoring disabled');
}

module.exports = Sentry; 