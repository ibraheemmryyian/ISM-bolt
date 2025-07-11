const { Queue } = require('bullmq');

// Redis connection configuration
const redisConfig = {
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
};

// Create the main AI generation queue
const aiGenerationQueue = new Queue('AIGenerationQueue', {
  connection: redisConfig,
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
    removeOnComplete: 100,
    removeOnFail: 50,
  },
});

// Create a queue for material listing generation
const materialListingQueue = new Queue('MaterialListingQueue', {
  connection: redisConfig,
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 3000,
    },
    removeOnComplete: 100,
    removeOnFail: 50,
  },
});

// Create a queue for company onboarding tasks
const onboardingQueue = new Queue('OnboardingQueue', {
  connection: redisConfig,
  defaultJobOptions: {
    attempts: 2,
    backoff: {
      type: 'exponential',
      delay: 5000,
    },
    removeOnComplete: 50,
    removeOnFail: 25,
  },
});

// Helper function to add AI generation job
async function addAIGenerationJob(companyData) {
  try {
    const job = await aiGenerationQueue.add('generate-listings', {
      companyId: companyData.companyId || companyData.id,
      companyName: companyData.name,
      industry: companyData.industry,
      location: companyData.location,
      materials: companyData.materials,
      processes: companyData.processes,
      wasteMaterials: companyData.waste_materials,
      timestamp: new Date().toISOString(),
    }, {
      priority: 1,
      delay: 1000, // 1 second delay to ensure company is saved first
    });

    console.log(`AI Generation job added: ${job.id} for company ${companyData.name}`);
    return job;
  } catch (error) {
    console.error('Error adding AI generation job:', error);
    throw error;
  }
}

// Helper function to add material listing job
async function addMaterialListingJob(companyData) {
  try {
    const job = await materialListingQueue.add('generate-materials', {
      companyId: companyData.companyId || companyData.id,
      companyName: companyData.name,
      industry: companyData.industry,
      location: companyData.location,
      volume: companyData.volume,
      materials: companyData.materials,
      wasteMaterials: companyData.waste_materials,
      timestamp: new Date().toISOString(),
    }, {
      priority: 2,
    });

    console.log(`Material Listing job added: ${job.id} for company ${companyData.name}`);
    return job;
  } catch (error) {
    console.error('Error adding material listing job:', error);
    throw error;
  }
}

// Helper function to add onboarding job
async function addOnboardingJob(companyData) {
  try {
    const job = await onboardingQueue.add('complete-onboarding', {
      companyId: companyData.companyId || companyData.id,
      companyName: companyData.name,
      username: companyData.username,
      email: companyData.email,
      industry: companyData.industry,
      timestamp: new Date().toISOString(),
    }, {
      priority: 1,
    });

    console.log(`Onboarding job added: ${job.id} for company ${companyData.name}`);
    return job;
  } catch (error) {
    console.error('Error adding onboarding job:', error);
    throw error;
  }
}

// Get queue statistics
async function getQueueStats() {
  try {
    const [aiStats, materialStats, onboardingStats] = await Promise.all([
      aiGenerationQueue.getJobCounts(),
      materialListingQueue.getJobCounts(),
      onboardingQueue.getJobCounts(),
    ]);

    return {
      aiGeneration: aiStats,
      materialListing: materialStats,
      onboarding: onboardingStats,
    };
  } catch (error) {
    console.error('Error getting queue stats:', error);
    return null;
  }
}

// Clean up completed jobs
async function cleanupQueues() {
  try {
    await Promise.all([
      aiGenerationQueue.clean(1000 * 60 * 60 * 24, 'completed'), // 24 hours
      materialListingQueue.clean(1000 * 60 * 60 * 24, 'completed'),
      onboardingQueue.clean(1000 * 60 * 60 * 24, 'completed'),
    ]);
    console.log('Queue cleanup completed');
  } catch (error) {
    console.error('Error cleaning up queues:', error);
  }
}

module.exports = {
  aiGenerationQueue,
  materialListingQueue,
  onboardingQueue,
  addAIGenerationJob,
  addMaterialListingJob,
  addOnboardingJob,
  getQueueStats,
  cleanupQueues,
}; 