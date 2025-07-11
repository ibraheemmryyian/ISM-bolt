const axios = require('axios');
const { supabase } = require('../supabase');

class HeightService {
  constructor() {
    this.apiKey = process.env.HEIGHT_API_KEY;
    this.baseUrl = 'https://api.height.app';
  }

  /**
   * Create a new project in Height for material exchange tracking
   */
  async createProject(projectData) {
    try {
      const response = await axios.post(`${this.baseUrl}/projects`, {
        name: projectData.name,
        description: projectData.description,
        workspaceId: process.env.HEIGHT_WORKSPACE_ID,
        color: projectData.color || '#3B82F6'
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error creating Height project:', error);
      throw error;
    }
  }

  /**
   * Create a task for material exchange tracking
   */
  async createTask(taskData) {
    try {
      const response = await axios.post(`${this.baseUrl}/tasks`, {
        name: taskData.name,
        description: taskData.description,
        projectId: taskData.projectId,
        assigneeId: taskData.assigneeId,
        dueDate: taskData.dueDate,
        priority: taskData.priority || 'medium',
        tags: taskData.tags || ['material-exchange', 'sustainability']
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error creating Height task:', error);
      throw error;
    }
  }

  /**
   * Create a material exchange tracking project and tasks
   */
  async createMaterialExchangeTracking(exchangeData) {
    try {
      // Create project for the exchange
      const project = await this.createProject({
        name: `Material Exchange: ${exchangeData.material_name}`,
        description: `Tracking material exchange between ${exchangeData.from_company} and ${exchangeData.to_company}`,
        color: '#10B981' // Green for sustainability
      });

      // Create tasks for different stages
      const tasks = [];

      // Task 1: Initial Setup
      const setupTask = await this.createTask({
        name: 'Exchange Setup & Validation',
        description: `Validate material specifications and company requirements for ${exchangeData.material_name}`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days from now
        priority: 'high',
        tags: ['setup', 'validation']
      });
      tasks.push(setupTask);

      // Task 2: Shipping Coordination
      const shippingTask = await this.createTask({
        name: 'Shipping & Logistics',
        description: `Coordinate shipping for ${exchangeData.quantity} units of ${exchangeData.material_name}`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days from now
        priority: 'medium',
        tags: ['shipping', 'logistics']
      });
      tasks.push(shippingTask);

      // Task 3: Quality Assurance
      const qaTask = await this.createTask({
        name: 'Quality Assurance & Inspection',
        description: `Quality check and inspection of received ${exchangeData.material_name}`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days from now
        priority: 'high',
        tags: ['quality', 'inspection']
      });
      tasks.push(qaTask);

      // Task 4: Documentation
      const docTask = await this.createTask({
        name: 'Documentation & Reporting',
        description: `Complete documentation and sustainability impact reporting for ${exchangeData.material_name} exchange`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toISOString(), // 10 days from now
        priority: 'medium',
        tags: ['documentation', 'reporting']
      });
      tasks.push(docTask);

      // Save Height project reference to database
      const { error } = await supabase
        .from('material_exchanges')
        .update({
          height_project_id: project.id,
          height_tasks: tasks.map(task => ({
            id: task.id,
            name: task.name,
            status: task.status
          }))
        })
        .eq('id', exchangeData.exchange_id);

      if (error) throw error;

      return {
        project_id: project.id,
        tasks: tasks,
        exchange_id: exchangeData.exchange_id
      };
    } catch (error) {
      console.error('Error creating Height tracking:', error);
      throw error;
    }
  }

  /**
   * Update task status
   */
  async updateTaskStatus(taskId, status) {
    try {
      const response = await axios.patch(`${this.baseUrl}/tasks/${taskId}`, {
        status: status
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error updating Height task status:', error);
      throw error;
    }
  }

  /**
   * Get project details and tasks
   */
  async getProjectDetails(projectId) {
    try {
      const [projectResponse, tasksResponse] = await Promise.all([
        axios.get(`${this.baseUrl}/projects/${projectId}`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`
          }
        }),
        axios.get(`${this.baseUrl}/tasks`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`
          },
          params: {
            projectId: projectId
          }
        })
      ]);

      return {
        project: projectResponse.data,
        tasks: tasksResponse.data
      };
    } catch (error) {
      console.error('Error fetching Height project details:', error);
      throw error;
    }
  }

  /**
   * Create a sustainability impact tracking project
   */
  async createSustainabilityTracking(impactData) {
    try {
      const project = await this.createProject({
        name: `Sustainability Impact: ${impactData.company_name}`,
        description: `Track sustainability metrics and environmental impact for ${impactData.company_name}`,
        color: '#059669' // Dark green for sustainability
      });

      const tasks = [];

      // CO2 Reduction Tracking
      const co2Task = await this.createTask({
        name: 'CO2 Reduction Monitoring',
        description: `Monitor and report CO2 reduction achievements for ${impactData.company_name}`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days
        priority: 'high',
        tags: ['co2', 'monitoring', 'sustainability']
      });
      tasks.push(co2Task);

      // Waste Diversion Tracking
      const wasteTask = await this.createTask({
        name: 'Waste Diversion Tracking',
        description: `Track landfill diversion and circular economy achievements`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        priority: 'high',
        tags: ['waste', 'diversion', 'circular-economy']
      });
      tasks.push(wasteTask);

      // Financial Impact Tracking
      const financialTask = await this.createTask({
        name: 'Financial Impact Analysis',
        description: `Analyze cost savings and ROI from sustainability initiatives`,
        projectId: project.id,
        dueDate: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000).toISOString(), // 45 days
        priority: 'medium',
        tags: ['financial', 'roi', 'analysis']
      });
      tasks.push(financialTask);

      return {
        project_id: project.id,
        tasks: tasks,
        company_name: impactData.company_name
      };
    } catch (error) {
      console.error('Error creating sustainability tracking:', error);
      throw error;
    }
  }

  /**
   * Get workspace members for assignment
   */
  async getWorkspaceMembers() {
    try {
      const response = await axios.get(`${this.baseUrl}/workspaces/${process.env.HEIGHT_WORKSPACE_ID}/members`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching Height workspace members:', error);
      throw error;
    }
  }

  /**
   * Create a comment on a task
   */
  async addTaskComment(taskId, comment) {
    try {
      const response = await axios.post(`${this.baseUrl}/tasks/${taskId}/comments`, {
        content: comment
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error adding Height task comment:', error);
      throw error;
    }
  }
}

module.exports = new HeightService(); 