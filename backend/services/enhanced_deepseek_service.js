const axios = require('axios');
const { EventEmitter } = require('events');

class EnhancedDeepSeekService extends EventEmitter {
  constructor() {
    super();
    this.apiKey = process.env.DEEPSEEK_API_KEY;
    this.baseUrl = 'https://api.deepseek.com/v1';
    this.model = 'deepseek-r1';
    
    if (!this.apiKey) {
      throw new Error('❌ DeepSeek R1 API key required for advanced AI reasoning');
    }
    
    // Advanced configuration
    this.config = {
      maxTokens: 4000,
      temperature: 0.1,
      topP: 0.9,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
      timeout: 30000
    };
    
    // Learning system
    this.interactionHistory = new Map();
    this.promptTemplates = new Map();
    this.performanceMetrics = new Map();
    
    // Initialize advanced prompt templates
    this.initializePromptTemplates();
    
    console.log('🚀 Enhanced DeepSeek R1 Service initialized with advanced AI capabilities');
  }

  /**
   * Initialize advanced prompt templates for industrial symbiosis
   */
  initializePromptTemplates() {
    // Multi-dimensional analysis templates
    this.promptTemplates.set('symbiosis_analysis', {
      system: `You are an expert industrial symbiosis analyst with deep knowledge of:
- Circular economy principles and implementation
- Material flow analysis and optimization
- Economic feasibility assessment
- Environmental impact calculation
- Regulatory compliance requirements
- Risk assessment and mitigation strategies
- Market dynamics and competitive analysis
- Technology transfer and innovation pathways

Analyze each request comprehensively using real data and provide actionable insights with confidence scores.`,
      
      user: `Analyze the following industrial symbiosis opportunity:

Company A: {companyA_data}
Company B: {companyB_data}
Material/Resource: {material_data}
Context: {context}

Provide a structured analysis covering:
1. Symbiosis Potential (0-100 score with reasoning)
2. Economic Viability (ROI, cost savings, revenue potential)
3. Environmental Impact (CO2 reduction, waste diversion, resource efficiency)
4. Technical Feasibility (processing requirements, compatibility, scalability)
5. Regulatory Compliance (permits, standards, certifications needed)
6. Risk Assessment (technical, financial, operational, market risks)
7. Implementation Roadmap (timeline, milestones, resource requirements)
8. Competitive Advantages (unique benefits, market positioning)
9. Success Metrics (KPIs, measurement methods, targets)
10. Recommendations (next steps, optimization opportunities)

Format response as structured JSON with detailed reasoning for each section.`
    });

    this.promptTemplates.set('market_intelligence', {
      system: `You are a market intelligence expert specializing in industrial markets, supply chains, and sustainability trends.`,
      user: `Analyze market intelligence for: {query}
Provide: market size, trends, competitors, opportunities, risks, and strategic recommendations.`
    });

    this.promptTemplates.set('logistics_optimization', {
      system: `You are a logistics optimization expert with expertise in sustainable transportation and supply chain efficiency.`,
      user: `Optimize logistics for: {logistics_data}
Consider: cost, time, carbon footprint, reliability, and sustainability.`
    });
  }

  /**
   * Advanced AI reasoning with multi-modal capabilities
   */
  async advancedReasoning(promptType, data, context = {}) {
    const tracking = this.trackRequest(promptType);
    
    try {
      // Get optimized prompt template
      const template = this.promptTemplates.get(promptType);
      if (!template) {
        throw new Error(`Unknown prompt type: ${promptType}`);
      }

      // Enhance prompt with context and learning
      const enhancedPrompt = this.enhancePrompt(template, data, context);
      
      // Multi-stage reasoning process
      const reasoning = await this.multiStageReasoning(enhancedPrompt, context);
      
      // Learn from interaction
      this.learnFromInteraction(promptType, data, reasoning, context);
      
      tracking.success();
      return reasoning;
      
    } catch (error) {
      tracking.error(error.message);
      throw new Error(`Advanced reasoning failed: ${error.message}`);
    }
  }

  /**
   * Multi-stage reasoning for complex analysis
   */
  async multiStageReasoning(prompt, context) {
    const stages = [
      'initial_analysis',
      'deep_dive',
      'validation',
      'synthesis'
    ];
    
    let currentResult = null;
    
    for (const stage of stages) {
      const stagePrompt = this.buildStagePrompt(prompt, stage, currentResult, context);
      const stageResult = await this.callDeepSeekAPI(stagePrompt, context);
      
      if (stageResult) {
        currentResult = this.mergeResults(currentResult, stageResult, stage);
      }
    }
    
    return currentResult;
  }

  /**
   * Build stage-specific prompts
   */
  buildStagePrompt(basePrompt, stage, previousResult, context) {
    const stagePrompts = {
      initial_analysis: `${basePrompt}\n\nStage 1: Perform initial analysis and identify key factors.`,
      deep_dive: `${basePrompt}\n\nStage 2: Deep dive into identified factors. Previous analysis: ${JSON.stringify(previousResult)}`,
      validation: `${basePrompt}\n\nStage 3: Validate findings and check for inconsistencies. Current analysis: ${JSON.stringify(previousResult)}`,
      synthesis: `${basePrompt}\n\nStage 4: Synthesize all findings into final recommendations. Complete analysis: ${JSON.stringify(previousResult)}`
    };
    
    return stagePrompts[stage] || basePrompt;
  }

  /**
   * Call DeepSeek API with advanced error handling
   */
  async callDeepSeekAPI(prompt, context = {}) {
    try {
      const messages = [
        {
          role: 'system',
          content: this.promptTemplates.get('symbiosis_analysis').system
        },
        {
          role: 'user',
          content: prompt
        }
      ];

      const response = await axios.post(
        `${this.baseUrl}/chat/completions`,
        {
          model: this.model,
          messages,
          max_tokens: this.config.maxTokens,
          temperature: this.config.temperature,
          top_p: this.config.topP,
          frequency_penalty: this.config.frequencyPenalty,
          presence_penalty: this.config.presencePenalty,
          stream: false
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: this.config.timeout
        }
      );

      const result = response.data.choices[0].message.content;
      
      // Try to parse JSON response
      try {
        return JSON.parse(result);
      } catch (parseError) {
        // If not JSON, return structured text
        return {
          analysis: result,
          confidence_score: this.calculateConfidence(result),
          timestamp: new Date().toISOString()
        };
      }
      
    } catch (error) {
      console.error('DeepSeek API error:', error.response?.data || error.message);
      throw new Error(`API call failed: ${error.message}`);
    }
  }

  /**
   * Enhance prompt with context and learning
   */
  enhancePrompt(template, data, context) {
    let enhancedPrompt = template.user;
    
    // Replace placeholders with actual data
    for (const [key, value] of Object.entries(data)) {
      enhancedPrompt = enhancedPrompt.replace(`{${key}}`, JSON.stringify(value));
    }
    
    // Add context information
    if (context.industry) {
      enhancedPrompt += `\n\nIndustry Context: ${context.industry}`;
    }
    
    if (context.location) {
      enhancedPrompt += `\n\nGeographic Context: ${context.location}`;
    }
    
    if (context.previousAnalysis) {
      enhancedPrompt += `\n\nPrevious Analysis: ${JSON.stringify(context.previousAnalysis)}`;
    }
    
    // Add learning from previous interactions
    const learningContext = this.getLearningContext(context);
    if (learningContext) {
      enhancedPrompt += `\n\nLearning Context: ${learningContext}`;
    }
    
    return enhancedPrompt;
  }

  /**
   * Learn from interactions for continuous improvement
   */
  learnFromInteraction(promptType, inputData, output, context) {
    const interactionKey = `${promptType}_${JSON.stringify(inputData).slice(0, 100)}`;
    
    this.interactionHistory.set(interactionKey, {
      input: inputData,
      output,
      context,
      timestamp: new Date().toISOString(),
      performance: this.calculatePerformance(output, context)
    });
    
    // Update performance metrics
    this.updatePerformanceMetrics(promptType, output, context);
    
    // Emit learning event
    this.emit('learning', {
      promptType,
      inputData,
      output,
      performance: this.calculatePerformance(output, context)
    });
  }

  /**
   * Get learning context from previous interactions
   */
  getLearningContext(context) {
    const relevantInteractions = [];
    
    for (const [key, interaction] of this.interactionHistory) {
      if (this.isRelevantInteraction(interaction, context)) {
        relevantInteractions.push(interaction);
      }
    }
    
    if (relevantInteractions.length === 0) {
      return null;
    }
    
    // Return most relevant learning insights
    return this.extractLearningInsights(relevantInteractions);
  }

  /**
   * Check if interaction is relevant to current context
   */
  isRelevantInteraction(interaction, context) {
    // Check industry relevance
    if (context.industry && interaction.context.industry) {
      if (context.industry !== interaction.context.industry) {
        return false;
      }
    }
    
    // Check location relevance
    if (context.location && interaction.context.location) {
      if (context.location !== interaction.context.location) {
        return false;
      }
    }
    
    // Check performance threshold
    return interaction.performance > 0.7;
  }

  /**
   * Extract learning insights from relevant interactions
   */
  extractLearningInsights(interactions) {
    const insights = {
      successful_patterns: [],
      common_factors: [],
      performance_trends: []
    };
    
    // Analyze successful patterns
    const successfulInteractions = interactions.filter(i => i.performance > 0.8);
    if (successfulInteractions.length > 0) {
      insights.successful_patterns = this.identifyPatterns(successfulInteractions);
    }
    
    // Identify common factors
    insights.common_factors = this.identifyCommonFactors(interactions);
    
    // Analyze performance trends
    insights.performance_trends = this.analyzePerformanceTrends(interactions);
    
    return insights;
  }

  /**
   * Calculate performance score for output
   */
  calculatePerformance(output, context) {
    let score = 0.5; // Base score
    
    // Check for structured response
    if (output && typeof output === 'object') {
      score += 0.2;
    }
    
    // Check for confidence score
    if (output.confidence_score) {
      score += output.confidence_score * 0.2;
    }
    
    // Check for completeness
    if (output.analysis && output.analysis.length > 100) {
      score += 0.1;
    }
    
    // Check for actionable insights
    if (output.recommendations || output.next_steps) {
      score += 0.2;
    }
    
    return Math.min(score, 1.0);
  }

  /**
   * Calculate confidence score for text response
   */
  calculateConfidence(text) {
    let confidence = 0.5; // Base confidence
    
    // Check for structured language
    if (text.includes('score:') || text.includes('confidence:')) {
      confidence += 0.2;
    }
    
    // Check for detailed reasoning
    if (text.split('.').length > 5) {
      confidence += 0.2;
    }
    
    // Check for specific metrics
    if (text.match(/\d+%/) || text.match(/\$\d+/)) {
      confidence += 0.1;
    }
    
    return Math.min(confidence, 1.0);
  }

  /**
   * Merge results from different stages
   */
  mergeResults(previous, current, stage) {
    if (!previous) {
      return current;
    }
    
    // Deep merge objects
    if (typeof current === 'object' && typeof previous === 'object') {
      return {
        ...previous,
        ...current,
        analysis_stages: {
          ...(previous.analysis_stages || {}),
          [stage]: current
        }
      };
    }
    
    return current;
  }

  /**
   * Track request performance
   */
  trackRequest(promptType) {
    const startTime = Date.now();
    
    return {
      success: () => {
        const duration = Date.now() - startTime;
        this.recordPerformance(promptType, duration, true);
      },
      error: (error) => {
        const duration = Date.now() - startTime;
        this.recordPerformance(promptType, duration, false, error);
      }
    };
  }

  /**
   * Record performance metrics
   */
  recordPerformance(promptType, duration, success, error = null) {
    if (!this.performanceMetrics.has(promptType)) {
      this.performanceMetrics.set(promptType, {
        total_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        average_duration: 0,
        errors: []
      });
    }
    
    const metrics = this.performanceMetrics.get(promptType);
    metrics.total_requests++;
    metrics.average_duration = (metrics.average_duration + duration) / 2;
    
    if (success) {
      metrics.successful_requests++;
    } else {
      metrics.failed_requests++;
      if (error) {
        metrics.errors.push(error);
      }
    }
  }

  /**
   * Update performance metrics
   */
  updatePerformanceMetrics(promptType, output, context) {
    const performance = this.calculatePerformance(output, context);
    
    if (!this.performanceMetrics.has(promptType)) {
      this.performanceMetrics.set(promptType, {
        average_performance: 0,
        performance_count: 0
      });
    }
    
    const metrics = this.performanceMetrics.get(promptType);
    metrics.performance_count++;
    metrics.average_performance = (metrics.average_performance + performance) / 2;
  }

  /**
   * Get performance analytics
   */
  getPerformanceAnalytics() {
    const analytics = {};
    
    for (const [promptType, metrics] of this.performanceMetrics) {
      analytics[promptType] = {
        ...metrics,
        success_rate: metrics.total_requests > 0 ? 
          (metrics.successful_requests / metrics.total_requests) * 100 : 0
      };
    }
    
    return analytics;
  }

  /**
   * Identify patterns in successful interactions
   */
  identifyPatterns(interactions) {
    const patterns = [];
    
    // Analyze input patterns
    const inputPatterns = this.analyzeInputPatterns(interactions);
    if (inputPatterns.length > 0) {
      patterns.push({ type: 'input_patterns', data: inputPatterns });
    }
    
    // Analyze output patterns
    const outputPatterns = this.analyzeOutputPatterns(interactions);
    if (outputPatterns.length > 0) {
      patterns.push({ type: 'output_patterns', data: outputPatterns });
    }
    
    return patterns;
  }

  /**
   * Analyze input patterns
   */
  analyzeInputPatterns(interactions) {
    const patterns = [];
    
    // Check for common data structures
    const dataTypes = new Set();
    interactions.forEach(i => {
      if (i.input && typeof i.input === 'object') {
        dataTypes.add(Object.keys(i.input).join(','));
      }
    });
    
    if (dataTypes.size > 0) {
      patterns.push({
        type: 'data_structure',
        description: 'Common input data structures',
        patterns: Array.from(dataTypes)
      });
    }
    
    return patterns;
  }

  /**
   * Analyze output patterns
   */
  analyzeOutputPatterns(interactions) {
    const patterns = [];
    
    // Check for common output fields
    const outputFields = new Set();
    interactions.forEach(i => {
      if (i.output && typeof i.output === 'object') {
        Object.keys(i.output).forEach(key => outputFields.add(key));
      }
    });
    
    if (outputFields.size > 0) {
      patterns.push({
        type: 'output_fields',
        description: 'Common output fields',
        fields: Array.from(outputFields)
      });
    }
    
    return patterns;
  }

  /**
   * Identify common factors
   */
  identifyCommonFactors(interactions) {
    const factors = {
      industries: new Set(),
      locations: new Set(),
      material_types: new Set()
    };
    
    interactions.forEach(i => {
      if (i.context.industry) factors.industries.add(i.context.industry);
      if (i.context.location) factors.locations.add(i.context.location);
      if (i.input.material_type) factors.material_types.add(i.input.material_type);
    });
    
    return {
      industries: Array.from(factors.industries),
      locations: Array.from(factors.locations),
      material_types: Array.from(factors.material_types)
    };
  }

  /**
   * Analyze performance trends
   */
  analyzePerformanceTrends(interactions) {
    if (interactions.length < 2) return [];
    
    const sortedInteractions = interactions.sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );
    
    const trends = [];
    let previousPerformance = sortedInteractions[0].performance;
    
    for (let i = 1; i < sortedInteractions.length; i++) {
      const currentPerformance = sortedInteractions[i].performance;
      const change = currentPerformance - previousPerformance;
      
      if (Math.abs(change) > 0.1) {
        trends.push({
          timestamp: sortedInteractions[i].timestamp,
          change,
          direction: change > 0 ? 'improving' : 'declining'
        });
      }
      
      previousPerformance = currentPerformance;
    }
    
    return trends;
  }

  /**
   * Get system health status
   */
  getHealthStatus() {
    const analytics = this.getPerformanceAnalytics();
    const totalRequests = Object.values(analytics).reduce((sum, a) => sum + a.total_requests, 0);
    const averageSuccessRate = Object.values(analytics).reduce((sum, a) => sum + a.success_rate, 0) / Object.keys(analytics).length;
    
    return {
      status: averageSuccessRate > 90 ? 'excellent' : averageSuccessRate > 80 ? 'good' : 'needs_attention',
      total_requests: totalRequests,
      average_success_rate: averageSuccessRate,
      interaction_history_size: this.interactionHistory.size,
      prompt_templates_count: this.promptTemplates.size,
      performance_metrics: analytics
    };
  }
}

module.exports = EnhancedDeepSeekService;