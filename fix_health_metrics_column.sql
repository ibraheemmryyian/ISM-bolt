-- Migration: Add metric_value column to health_metrics if missing
ALTER TABLE health_metrics ADD COLUMN IF NOT EXISTS metric_value REAL; 