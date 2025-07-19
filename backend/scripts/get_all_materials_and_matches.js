// Load environment variables from .env if available
try {
  require('dotenv').config({ path: require('path').resolve(__dirname, '../../.env') });
} catch (e) {
  // dotenv not installed, ignore
}

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your environment or .env file.');
  process.exit(1);
}

const companies = JSON.parse(fs.readFileSync(path.join(__dirname, '../../fixed_realworlddata.json'), 'utf-8'));
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function getListingsAndMatches() {
  const results = [];
  for (const company of companies) {
    // Find company in DB by name (case-insensitive)
    const { data: dbCompany, error: companyError } = await supabase
      .from('companies')
      .select('id, name')
      .ilike('name', company.name);
    if (!dbCompany || dbCompany.length === 0) {
      results.push({ name: company.name, found: false, listings: 0, matches: 0 });
      continue;
    }
    const companyId = dbCompany[0].id;
    // Get material listings
    const { data: materials, error: materialsError } = await supabase
      .from('materials')
      .select('*')
      .eq('company_id', companyId);
    // Get matches (if you have a matches table)
    const { data: matches, error: matchesError } = await supabase
      .from('matches')
      .select('*')
      .eq('company_id', companyId);
    results.push({
      name: company.name,
      found: true,
      listings: materials ? materials.length : 0,
      matches: matches ? matches.length : 0,
      materialDetails: materials || [],
      matchDetails: matches || []
    });
  }
  // Write summary report
  fs.writeFileSync(path.join(__dirname, 'material_match_report.json'), JSON.stringify(results, null, 2));
  console.log('Report written to backend/scripts/material_match_report.json');
  // Print summary
  results.forEach(r => {
    console.log(`${r.name}: Listings=${r.listings}, Matches=${r.matches}`);
  });
}

getListingsAndMatches(); 