import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Play, 
  Pause, 
  MessageSquare,
  ExternalLink,
  Loader2,
  Calendar,
  User,
  Tag
} from 'lucide-react';

interface HeightTask {
  id: string;
  name: string;
  description: string;
  status: string;
  priority: string;
  dueDate: string;
  assigneeId?: string;
  tags: string[];
}

interface HeightProject {
  id: string;
  name: string;
  description: string;
  status: string;
  tasks: HeightTask[];
}

interface HeightProjectTrackerProps {
  projectId?: string;
  exchangeId?: string;
  companyId?: string;
  projectType: 'material_exchange' | 'sustainability' | 'implementation';
  onProjectCreated?: (project: HeightProject) => void;
}

const HeightProjectTracker: React.FC<HeightProjectTrackerProps> = ({
  projectId,
  exchangeId,
  companyId,
  projectType,
  onProjectCreated
}) => {
  const [project, setProject] = useState<HeightProject | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newComment, setNewComment] = useState('');
  const [commentingTaskId, setCommentingTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (projectId) {
      fetchProjectDetails();
    }
  }, [projectId]);

  const fetchProjectDetails = async () => {
    if (!projectId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/height/project/${projectId}`);
      const data = await response.json();

      if (data.success) {
        setProject(data.project);
      } else {
        setError(data.error || 'Failed to fetch project details');
      }
    } catch (error) {
      console.error('Error fetching project details:', error);
      setError('Failed to fetch project details');
    } finally {
      setLoading(false);
    }
  };

  const createMaterialExchangeTracking = async (exchangeData: any) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/height/create-exchange-tracking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ exchangeData })
      });

      const data = await response.json();

      if (data.success) {
        setProject({
          id: data.tracking.project_id,
          name: `Material Exchange: ${exchangeData.material_name}`,
          description: `Tracking material exchange between ${exchangeData.from_company} and ${exchangeData.to_company}`,
          status: 'active',
          tasks: data.tracking.tasks
        });
        onProjectCreated?.(data.tracking);
      } else {
        setError(data.error || 'Failed to create tracking project');
      }
    } catch (error) {
      console.error('Error creating tracking project:', error);
      setError('Failed to create tracking project');
    } finally {
      setLoading(false);
    }
  };

  const createSustainabilityTracking = async (impactData: any) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/height/create-sustainability-tracking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ impactData })
      });

      const data = await response.json();

      if (data.success) {
        setProject({
          id: data.tracking.project_id,
          name: `Sustainability Impact: ${impactData.company_name}`,
          description: `Track sustainability metrics and environmental impact for ${impactData.company_name}`,
          status: 'active',
          tasks: data.tracking.tasks
        });
        onProjectCreated?.(data.tracking);
      } else {
        setError(data.error || 'Failed to create sustainability tracking');
      }
    } catch (error) {
      console.error('Error creating sustainability tracking:', error);
      setError('Failed to create sustainability tracking');
    } finally {
      setLoading(false);
    }
  };

  const updateTaskStatus = async (taskId: string, status: string) => {
    try {
      const response = await fetch(`/api/height/task/${taskId}/status`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status })
      });

      const data = await response.json();

      if (data.success) {
        // Update the task in the local state
        setProject(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            tasks: prev.tasks.map(task =>
              task.id === taskId ? { ...task, status } : task
            )
          };
        });
      } else {
        setError(data.error || 'Failed to update task status');
      }
    } catch (error) {
      console.error('Error updating task status:', error);
      setError('Failed to update task status');
    }
  };

  const addTaskComment = async (taskId: string, comment: string) => {
    if (!comment.trim()) return;

    try {
      const response = await fetch(`/api/height/task/${taskId}/comment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ comment })
      });

      const data = await response.json();

      if (data.success) {
        setNewComment('');
        setCommentingTaskId(null);
        // Optionally refresh project details to get the new comment
        await fetchProjectDetails();
      } else {
        setError(data.error || 'Failed to add comment');
      }
    } catch (error) {
      console.error('Error adding comment:', error);
      setError('Failed to add comment');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'in_progress':
        return <Play className="w-4 h-4 text-blue-500" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'blocked':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const openInHeight = (projectId: string) => {
    window.open(`https://height.app/project/${projectId}`, '_blank');
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center p-6">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading project details...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center text-red-600">
            <AlertCircle className="w-5 h-5 mr-2" />
            <span>{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!project) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <ExternalLink className="w-5 h-5 mr-2" />
            Height Project Tracking
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600 mb-4">
            No project tracking found. Create a new Height project to track this {projectType.replace('_', ' ')}.
          </p>
          <Button 
            onClick={() => {
              if (projectType === 'material_exchange' && exchangeId) {
                // This would need exchange data from parent component
                console.log('Create material exchange tracking');
              } else if (projectType === 'sustainability' && companyId) {
                createSustainabilityTracking({ company_id: companyId, company_name: 'Your Company' });
              }
            }}
            disabled={loading}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
            Create Height Project
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <ExternalLink className="w-5 h-5 mr-2" />
            {project.name}
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => openInHeight(project.id)}
          >
            <ExternalLink className="w-4 h-4 mr-2" />
            Open in Height
          </Button>
        </div>
        <p className="text-gray-600">{project.description}</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {project.tasks.map((task) => (
            <div key={task.id} className="border rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(task.status)}
                  <h4 className="font-medium">{task.name}</h4>
                  <Badge className={getPriorityColor(task.priority)}>
                    {task.priority}
                  </Badge>
                </div>
                <div className="flex items-center space-x-2">
                  {task.dueDate && (
                    <div className="flex items-center text-sm text-gray-500">
                      <Calendar className="w-4 h-4 mr-1" />
                      {new Date(task.dueDate).toLocaleDateString()}
                    </div>
                  )}
                  {task.assigneeId && (
                    <div className="flex items-center text-sm text-gray-500">
                      <User className="w-4 h-4 mr-1" />
                      Assigned
                    </div>
                  )}
                </div>
              </div>
              
              <p className="text-gray-600 text-sm mb-3">{task.description}</p>
              
              {task.tags && task.tags.length > 0 && (
                <div className="flex items-center space-x-2 mb-3">
                  <Tag className="w-4 h-4 text-gray-400" />
                  {task.tags.map((tag, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <select
                    value={task.status}
                    onChange={(e) => updateTaskStatus(task.id, e.target.value)}
                    className="text-sm border rounded px-2 py-1"
                  >
                    <option value="pending">Pending</option>
                    <option value="in_progress">In Progress</option>
                    <option value="completed">Completed</option>
                    <option value="blocked">Blocked</option>
                  </select>
                </div>
                
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCommentingTaskId(commentingTaskId === task.id ? null : task.id)}
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Comment
                </Button>
              </div>
              
              {commentingTaskId === task.id && (
                <div className="mt-3 space-y-2">
                  <textarea
                    value={newComment}
                    onChange={(e) => setNewComment(e.target.value)}
                    placeholder="Add a comment..."
                    className="w-full p-2 border rounded text-sm"
                    rows={2}
                  />
                  <div className="flex items-center space-x-2">
                    <Button
                      size="sm"
                      onClick={() => addTaskComment(task.id, newComment)}
                      disabled={!newComment.trim()}
                    >
                      Add Comment
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setNewComment('');
                        setCommentingTaskId(null);
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default HeightProjectTracker; 