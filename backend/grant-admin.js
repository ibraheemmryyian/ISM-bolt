const supabase = require('./supabase');

async function grantAdminAccess(email) {
  try {
    console.log(`Granting admin access to: ${email}`);
    
    const { data, error } = await supabase
      .from('companies')
      .update({ role: 'admin' })
      .eq('email', email)
      .select();

    if (error) {
      console.error('Error granting admin access:', error);
      return;
    }

    if (data && data.length > 0) {
      console.log('✅ Admin access granted successfully!');
      console.log('Updated user:', data[0]);
    } else {
      console.log('❌ No user found with that email address');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

// Get email from command line argument
const email = process.argv[2];
if (!email) {
  console.log('Usage: node grant-admin.js <email>');
  console.log('Example: node grant-admin.js admin@example.com');
  process.exit(1);
}

grantAdminAccess(email); 