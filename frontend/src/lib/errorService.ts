/**
 * Error logging service for centralized error handling and logging
 * NO FALLBACK LOGIC - FAILS LOUDLY TO EXPOSE CRITICAL FAULTS
 */

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error'
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  error?: Error;
  context?: Record<string, any>;
  timestamp: Date;
  userId?: string;
  component?: string;
}

class ErrorService {
  private logs: LogEntry[] = [];
  private maxLogs = 1000;
  private isProduction = import.meta.env.PROD;

  private createLogEntry(
    level: LogLevel,
    message: string,
    error?: Error,
    context?: Record<string, any>,
    component?: string
  ): LogEntry {
    return {
      level,
      message,
      error,
      context,
      timestamp: new Date(),
      userId: this.getCurrentUserId(),
      component
    };
  }

  private getCurrentUserId(): string | undefined {
    // NO FALLBACK - If this fails, it should throw
    const user = localStorage.getItem('sb-user');
    if (!user) {
      throw new Error('User data not found in localStorage - authentication may be broken');
    }
    
    try {
      const parsedUser = JSON.parse(user);
      if (!parsedUser.id) {
        throw new Error('User ID not found in parsed user data');
      }
      return parsedUser.id;
    } catch (parseError) {
      throw new Error(`Failed to parse user data: ${parseError instanceof Error ? parseError.message : 'Unknown parse error'}`);
    }
  }

  private addLog(entry: LogEntry): void {
    this.logs.push(entry);
    
    // Keep only the last maxLogs entries
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // In development, also log to console
    if (!this.isProduction) {
      const consoleMethod = entry.level === LogLevel.ERROR ? 'error' : 
                           entry.level === LogLevel.WARN ? 'warn' : 
                           entry.level === LogLevel.INFO ? 'info' : 'log';
      
      console[consoleMethod](
        `[${entry.component || 'App'}] ${entry.message}`,
        entry.error || '',
        entry.context || ''
      );
    }

    // In production, send to backend logging service - NO FALLBACK
    if (this.isProduction && entry.level === LogLevel.ERROR) {
      this.sendToBackend(entry);
    }
  }

  private async sendToBackend(entry: LogEntry): Promise<void> {
    // NO FALLBACK - If backend logging fails, throw error to expose the fault
    const response = await fetch('/api/logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry)
    });

    if (!response.ok) {
      throw new Error(`Backend logging failed with status ${response.status}: ${response.statusText}`);
    }
  }

  debug(message: string, context?: Record<string, any>, component?: string): void {
    this.addLog(this.createLogEntry(LogLevel.DEBUG, message, undefined, context, component));
  }

  info(message: string, context?: Record<string, any>, component?: string): void {
    this.addLog(this.createLogEntry(LogLevel.INFO, message, undefined, context, component));
  }

  warn(message: string, error?: Error, context?: Record<string, any>, component?: string): void {
    this.addLog(this.createLogEntry(LogLevel.WARN, message, error, context, component));
  }

  error(message: string, error?: Error, context?: Record<string, any>, component?: string): void {
    this.addLog(this.createLogEntry(LogLevel.ERROR, message, error, context, component));
  }

  // Convenience methods for common error patterns
  apiError(operation: string, error: Error, context?: Record<string, any>, component?: string): void {
    this.error(`API Error in ${operation}`, error, context, component);
  }

  authError(operation: string, error: Error, context?: Record<string, any>, component?: string): void {
    this.error(`Authentication Error in ${operation}`, error, context, component);
  }

  validationError(field: string, value: any, context?: Record<string, any>, component?: string): void {
    this.warn(`Validation Error: Invalid ${field}`, undefined, { field, value, ...context }, component);
  }

  // Get logs for debugging
  getLogs(): LogEntry[] {
    return [...this.logs];
  }

  // Clear logs
  clearLogs(): void {
    this.logs = [];
  }

  // Export logs
  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }
}

// Create singleton instance
export const errorService = new ErrorService();

// Convenience exports
export const log = {
  debug: (message: string, context?: Record<string, any>, component?: string) => 
    errorService.debug(message, context, component),
  info: (message: string, context?: Record<string, any>, component?: string) => 
    errorService.info(message, context, component),
  warn: (message: string, error?: Error, context?: Record<string, any>, component?: string) => 
    errorService.warn(message, error, context, component),
  error: (message: string, error?: Error, context?: Record<string, any>, component?: string) => 
    errorService.error(message, error, context, component),
  apiError: (operation: string, error: Error, context?: Record<string, any>, component?: string) => 
    errorService.apiError(operation, error, context, component),
  authError: (operation: string, error: Error, context?: Record<string, any>, component?: string) => 
    errorService.authError(operation, error, context, component),
  validationError: (field: string, value: any, context?: Record<string, any>, component?: string) => 
    errorService.validationError(field, value, context, component)
}; 