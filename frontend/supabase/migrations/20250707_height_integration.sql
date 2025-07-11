-- Height API Integration Migration
-- Adds project management tracking for material exchanges and sustainability initiatives

-- Add Height tracking columns to material_exchanges table
ALTER TABLE material_exchanges ADD COLUMN IF NOT EXISTS height_project_id varchar(100);
ALTER TABLE material_exchanges ADD COLUMN IF NOT EXISTS height_tasks jsonb;
ALTER TABLE material_exchanges ADD COLUMN IF NOT EXISTS project_tracking_status varchar(50) DEFAULT 'pending';

-- Add Height tracking columns to companies table for sustainability projects
ALTER TABLE companies ADD COLUMN IF NOT EXISTS height_sustainability_project_id varchar(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS height_sustainability_tasks jsonb;

-- Add Height tracking columns to ai_insights table
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS height_implementation_project_id varchar(100);
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS height_implementation_tasks jsonb;

-- Create Height project tracking table
CREATE TABLE IF NOT EXISTS height_projects (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id varchar(100) NOT NULL UNIQUE,
    project_name varchar(255) NOT NULL,
    project_type varchar(50) NOT NULL, -- 'material_exchange', 'sustainability', 'implementation'
    related_entity_id uuid, -- References material_exchanges.id, companies.id, or ai_insights.id
    related_entity_type varchar(50), -- 'material_exchange', 'company', 'ai_insight'
    workspace_id varchar(100),
    status varchar(50) DEFAULT 'active',
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    metadata jsonb
);

-- Create Height tasks tracking table
CREATE TABLE IF NOT EXISTS height_tasks (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id varchar(100) NOT NULL UNIQUE,
    project_id varchar(100) NOT NULL,
    task_name varchar(255) NOT NULL,
    task_description text,
    assignee_id varchar(100),
    due_date timestamp with time zone,
    priority varchar(20) DEFAULT 'medium',
    status varchar(50) DEFAULT 'pending',
    tags text[],
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    metadata jsonb,
    FOREIGN KEY (project_id) REFERENCES height_projects(project_id) ON DELETE CASCADE
);

-- Create Height comments tracking table
CREATE TABLE IF NOT EXISTS height_comments (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    comment_id varchar(100) NOT NULL UNIQUE,
    task_id varchar(100) NOT NULL,
    content text NOT NULL,
    author_id varchar(100),
    created_at timestamp with time zone DEFAULT now(),
    metadata jsonb,
    FOREIGN KEY (task_id) REFERENCES height_tasks(task_id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_height_projects_entity ON height_projects(related_entity_id, related_entity_type);
CREATE INDEX IF NOT EXISTS idx_height_projects_type ON height_projects(project_type);
CREATE INDEX IF NOT EXISTS idx_height_tasks_project ON height_tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_height_tasks_status ON height_tasks(status);
CREATE INDEX IF NOT EXISTS idx_height_comments_task ON height_comments(task_id);

-- Add RLS policies for Height tables
ALTER TABLE height_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE height_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE height_comments ENABLE ROW LEVEL SECURITY;

-- RLS policies for height_projects
CREATE POLICY "Users can view their own Height projects" ON height_projects
    FOR SELECT USING (
        related_entity_type = 'company' AND 
        related_entity_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert their own Height projects" ON height_projects
    FOR INSERT WITH CHECK (
        related_entity_type = 'company' AND 
        related_entity_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update their own Height projects" ON height_projects
    FOR UPDATE USING (
        related_entity_type = 'company' AND 
        related_entity_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

-- RLS policies for height_tasks
CREATE POLICY "Users can view tasks for their projects" ON height_tasks
    FOR SELECT USING (
        project_id IN (
            SELECT project_id FROM height_projects 
            WHERE related_entity_type = 'company' AND 
                  related_entity_id IN (
                      SELECT id FROM companies WHERE user_id = auth.uid()
                  )
        )
    );

CREATE POLICY "Users can insert tasks for their projects" ON height_tasks
    FOR INSERT WITH CHECK (
        project_id IN (
            SELECT project_id FROM height_projects 
            WHERE related_entity_type = 'company' AND 
                  related_entity_id IN (
                      SELECT id FROM companies WHERE user_id = auth.uid()
                  )
        )
    );

CREATE POLICY "Users can update tasks for their projects" ON height_tasks
    FOR UPDATE USING (
        project_id IN (
            SELECT project_id FROM height_projects 
            WHERE related_entity_type = 'company' AND 
                  related_entity_id IN (
                      SELECT id FROM companies WHERE user_id = auth.uid()
                  )
        )
    );

-- RLS policies for height_comments
CREATE POLICY "Users can view comments for their tasks" ON height_comments
    FOR SELECT USING (
        task_id IN (
            SELECT task_id FROM height_tasks 
            WHERE project_id IN (
                SELECT project_id FROM height_projects 
                WHERE related_entity_type = 'company' AND 
                      related_entity_id IN (
                          SELECT id FROM companies WHERE user_id = auth.uid()
                      )
            )
        )
    );

CREATE POLICY "Users can insert comments for their tasks" ON height_comments
    FOR INSERT WITH CHECK (
        task_id IN (
            SELECT task_id FROM height_tasks 
            WHERE project_id IN (
                SELECT project_id FROM height_projects 
                WHERE related_entity_type = 'company' AND 
                      related_entity_id IN (
                          SELECT id FROM companies WHERE user_id = auth.uid()
                      )
            )
        )
    );

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_height_projects_updated_at 
    BEFORE UPDATE ON height_projects 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_height_tasks_updated_at 
    BEFORE UPDATE ON height_tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE height_projects IS 'Tracks Height.app projects for material exchanges and sustainability initiatives';
COMMENT ON TABLE height_tasks IS 'Tracks Height.app tasks within projects';
COMMENT ON TABLE height_comments IS 'Tracks Height.app comments on tasks';
COMMENT ON COLUMN material_exchanges.height_project_id IS 'Height.app project ID for tracking this material exchange';
COMMENT ON COLUMN material_exchanges.height_tasks IS 'JSON array of Height.app task IDs and statuses for this exchange';
COMMENT ON COLUMN companies.height_sustainability_project_id IS 'Height.app project ID for tracking sustainability initiatives';
COMMENT ON COLUMN ai_insights.height_implementation_project_id IS 'Height.app project ID for tracking AI implementation'; 