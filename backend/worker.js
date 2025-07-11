const { Worker } = require('bullmq');
const { PythonShell } = require('python-shell');
const path = require('path');
const { supabase } = require('./supabase');

// Redis connection configuration
const redisConfig = {
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
};

// AI Generation Worker
const aiGenerationWorker = new Worker('AIGenerationQueue', async (job) => {
  console.log(`🔄 Processing AI Generation job ${job.id} for company: ${job.data.companyName}`);
  
  try {
    // Call the Python AI service
    const result = await callAIService(job.data);
    
    // Save the generated listings to the database
    await saveGeneratedListings(job.data.companyId, result);
    
    console.log(`✅ AI Generation job ${job.id} completed successfully`);
    return { success: true, listingsGenerated: result.length };
    
  } catch (error) {
    console.error(`❌ AI Generation job ${job.id} failed:`, error);
    throw error;
  }
}, {
  connection: redisConfig,
  concurrency: 2, // Process 2 jobs at a time
});

// Material Listing Worker
const materialListingWorker = new Worker('MaterialListingQueue', async (job) => {
  console.log(`🔄 Processing Material Listing job ${job.id} for company: ${job.data.companyName}`);
  
  try {
    // Generate material listings based on company profile
    const materials = await generateMaterialListings(job.data);
    
    // Save materials to database
    await saveMaterialListings(job.data.companyId, materials);
    
    console.log(`✅ Material Listing job ${job.id} completed successfully`);
    return { success: true, materialsGenerated: materials.length };
    
  } catch (error) {
    console.error(`❌ Material Listing job ${job.id} failed:`, error);
    throw error;
  }
}, {
  connection: redisConfig,
  concurrency: 3, // Process 3 jobs at a time
});

// Onboarding Worker
const onboardingWorker = new Worker('OnboardingQueue', async (job) => {
  console.log(`🔄 Processing Onboarding job ${job.id} for company: ${job.data.companyName}`);
  
  try {
    // Complete onboarding tasks
    await completeOnboarding(job.data);
    
    console.log(`✅ Onboarding job ${job.id} completed successfully`);
    return { success: true };
    
  } catch (error) {
    console.error(`❌ Onboarding job ${job.id} failed:`, error);
    throw error;
  }
}, {
  connection: redisConfig,
  concurrency: 1, // Process 1 job at a time
});

// Call Python AI service
async function callAIService(companyData) {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'json',
      pythonPath: 'python',
      pythonOptions: ['-u'], // unbuffered output
      scriptPath: __dirname,
      args: [
        '--company_data', JSON.stringify(companyData)
      ]
    };

    PythonShell.run('listing_inference_service.py', options, (err, results) => {
      if (err) {
        console.error('Python AI service error:', err);
        reject(err);
        return;
      }

      if (results && results.length > 0) {
        try {
          const result = JSON.parse(results[0]);
          resolve(result.listings || []);
        } catch (parseError) {
          console.error('Error parsing AI service result:', parseError);
          reject(parseError);
        }
      } else {
        resolve([]);
      }
    });
  });
}

// Generate material listings based on company profile
async function generateMaterialListings(companyData) {
  const materials = [];
  
  // Generate waste materials from waste_materials field
  if (companyData.wasteMaterials) {
    const wasteList = companyData.wasteMaterials.split(',').map(w => w.trim());
    
    for (const waste of wasteList) {
      if (waste) {
        const quantity = parseVolumeToQuantity(companyData.volume);
        const unit = getUnitFromVolume(companyData.volume);
        
        materials.push({
          material_name: waste,
          quantity: quantity,
          unit: unit,
          type: 'waste',
          description: `${waste} from ${companyData.companyName} ${companyData.industry} operations`,
          availability: 'Available',
          price_per_unit: Math.random() * 5 + 0.1,
          location: companyData.location,
          created_at: new Date().toISOString()
        });
      }
    }
  }
  
  // Generate requirement materials based on industry
  const requirementMaterials = getRequirementMaterialsForIndustry(companyData.industry);
  
  for (const reqMaterial of requirementMaterials) {
    materials.push({
      material_name: reqMaterial,
      quantity: Math.floor(Math.random() * 50000) + 500,
      unit: reqMaterial === 'Energy' ? 'kWh' : 'kg',
      type: 'requirement',
      description: `${reqMaterial} needed for ${companyData.companyName} operations`,
      availability: 'Needed',
      price_per_unit: Math.random() * 10 + 0.5,
      location: companyData.location,
      created_at: new Date().toISOString()
    });
  }
  
  return materials;
}

// Save generated listings to database
async function saveGeneratedListings(companyId, listings) {
  if (!listings || listings.length === 0) return;
  
  try {
    const { error } = await supabase
      .from('materials')
      .insert(listings.map(listing => ({
        ...listing,
        company_id: companyId
      })));
    
    if (error) throw error;
    
    console.log(`💾 Saved ${listings.length} listings for company ${companyId}`);
  } catch (error) {
    console.error('Error saving listings:', error);
    throw error;
  }
}

// Save material listings to database
async function saveMaterialListings(companyId, materials) {
  if (!materials || materials.length === 0) return;
  
  try {
    const { error } = await supabase
      .from('materials')
      .insert(materials.map(material => ({
        ...material,
        company_id: companyId
      })));
    
    if (error) throw error;
    
    console.log(`💾 Saved ${materials.length} materials for company ${companyId}`);
  } catch (error) {
    console.error('Error saving materials:', error);
    throw error;
  }
}

// Complete onboarding tasks
async function completeOnboarding(companyData) {
  try {
    // Update company status to onboarded
    const { error } = await supabase
      .from('companies')
      .update({ 
        onboarding_completed: true,
        onboarding_completed_at: new Date().toISOString()
      })
      .eq('id', companyData.companyId);
    
    if (error) throw error;
    
    console.log(`✅ Onboarding completed for company ${companyData.companyName}`);
  } catch (error) {
    console.error('Error completing onboarding:', error);
    throw error;
  }
}

// Helper functions
function parseVolumeToQuantity(volumeStr) {
  if (!volumeStr) return Math.floor(Math.random() * 10000) + 100;
  
  const numbers = volumeStr.match(/\d+/);
  if (numbers) {
    const baseQuantity = parseInt(numbers[0]);
    return Math.floor(baseQuantity * (0.8 + Math.random() * 0.4));
  }
  
  return Math.floor(Math.random() * 10000) + 100;
}

function getUnitFromVolume(volumeStr) {
  if (!volumeStr) return 'kg';
  
  if (volumeStr.toLowerCase().includes('cubic meters')) return 'm³';
  if (volumeStr.toLowerCase().includes('metric tons')) return 'kg';
  if (volumeStr.toLowerCase().includes('liters')) return 'L';
  
  return 'kg';
}

function getRequirementMaterialsForIndustry(industry) {
  const industryLower = industry?.toLowerCase() || '';
  
  const baseRequirements = ['Energy', 'Water', 'Packaging Materials'];
  
  if (industryLower.includes('construction') || industryLower.includes('real estate')) {
    return baseRequirements.concat(['Steel', 'Concrete', 'Glass', 'Wood', 'Aluminum']);
  }
  if (industryLower.includes('manufacturing')) {
    return baseRequirements.concat(['Raw Materials', 'Polymers', 'Fabrics', 'Agricultural Produce']);
  }
  if (industryLower.includes('oil') || industryLower.includes('gas')) {
    return baseRequirements.concat(['Drilling Fluids', 'Catalysts', 'Hydrocarbons', 'Pipelines']);
  }
  if (industryLower.includes('healthcare')) {
    return baseRequirements.concat(['Pharmaceuticals', 'Medical Supplies', 'Lab Reagents', 'Sterile Packaging']);
  }
  if (industryLower.includes('tourism') || industryLower.includes('hospitality')) {
    return baseRequirements.concat(['Food & Beverages', 'Linens', 'Cleaning Supplies', 'Paper Products']);
  }
  if (industryLower.includes('water treatment')) {
    return baseRequirements.concat(['Raw Water', 'Chemical Coagulants', 'Disinfectants', 'Filtration Media']);
  }
  if (industryLower.includes('logistics') || industryLower.includes('transportation')) {
    return baseRequirements.concat(['Fuel', 'Vehicle Parts', 'Packaging Materials']);
  }
  
  return baseRequirements.concat(['Raw Materials', 'Cleaning Supplies']);
}

// Worker event handlers
aiGenerationWorker.on('completed', (job) => {
  console.log(`🎉 AI Generation job ${job.id} completed successfully`);
});

aiGenerationWorker.on('failed', (job, err) => {
  console.error(`💥 AI Generation job ${job.id} failed:`, err.message);
});

materialListingWorker.on('completed', (job) => {
  console.log(`🎉 Material Listing job ${job.id} completed successfully`);
});

materialListingWorker.on('failed', (job, err) => {
  console.error(`💥 Material Listing job ${job.id} failed:`, err.message);
});

onboardingWorker.on('completed', (job) => {
  console.log(`🎉 Onboarding job ${job.id} completed successfully`);
});

onboardingWorker.on('failed', (job, err) => {
  console.error(`💥 Onboarding job ${job.id} failed:`, err.message);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down workers gracefully...');
  await Promise.all([
    aiGenerationWorker.close(),
    materialListingWorker.close(),
    onboardingWorker.close(),
  ]);
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('🛑 Shutting down workers gracefully...');
  await Promise.all([
    aiGenerationWorker.close(),
    materialListingWorker.close(),
    onboardingWorker.close(),
  ]);
  process.exit(0);
});

console.log('🚀 AI Workers are running and waiting for jobs...');
console.log('📊 AI Generation Worker: Ready');
console.log('📦 Material Listing Worker: Ready');
console.log('👋 Onboarding Worker: Ready'); 