-- Fix missing environment validation and monitoring tables
-- Run this in your Supabase SQL editor

-- Create environment validation table
CREATE TABLE IF NOT EXISTS environment_validation (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    validation_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create application logs table for monitoring
CREATE TABLE IF NOT EXISTS application_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    user_agent TEXT,
    url TEXT,
    source VARCHAR(20) DEFAULT 'frontend',
    user_id UUID REFERENCES auth.users(id)
);

-- Create error logs table
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    error_id VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    stack_trace TEXT,
    component_stack TEXT,
    user_agent TEXT,
    url TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source VARCHAR(20) DEFAULT 'frontend',
    user_id UUID REFERENCES auth.users(id)
);

-- Create system health monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_count INTEGER DEFAULT 0,
    last_check TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_application_logs_timestamp ON application_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_level ON application_logs(level);
CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_service ON system_health(service_name);

-- Insert initial system health records
INSERT INTO system_health (service_name, status, response_time_ms) VALUES
('frontend', 'healthy', 0),
('backend', 'healthy', 0),
('database', 'healthy', 0),
('ai_services', 'healthy', 0)
ON CONFLICT DO NOTHING;

-- Add RLS policies
ALTER TABLE environment_validation ENABLE ROW LEVEL SECURITY;
ALTER TABLE application_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE error_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;

-- Admin can read all logs
CREATE POLICY "Admin can read all logs" ON application_logs
    FOR SELECT USING (auth.jwt() ->> 'role' = 'admin');

CREATE POLICY "Admin can read all errors" ON error_logs
    FOR SELECT USING (auth.jwt() ->> 'role' = 'admin');

CREATE POLICY "Admin can read system health" ON system_health
    FOR SELECT USING (auth.jwt() ->> 'role' = 'admin');

-- Users can insert their own logs
CREATE POLICY "Users can insert their own logs" ON application_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can insert their own errors" ON error_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id); 