// Admin Access Utility
export const grantAdminAccess = () => {
  localStorage.setItem('temp-admin-access', 'true');
  localStorage.setItem('admin-user-id', 'admin-user');
  console.log('✅ Admin access granted!');
  return true;
};

export const revokeAdminAccess = () => {
  localStorage.removeItem('temp-admin-access');
  localStorage.removeItem('admin-user-id');
  console.log('❌ Admin access revoked!');
  return true;
};

export const checkAdminAccess = () => {
  const hasAccess = localStorage.getItem('temp-admin-access') === 'true';
  console.log('🔐 Admin access status:', hasAccess ? 'GRANTED' : 'DENIED');
  return hasAccess;
};

// Quick admin access for development
if (typeof window !== 'undefined') {
  // Add to window for easy access in console
  (window as any).grantAdmin = grantAdminAccess;
  (window as any).revokeAdmin = revokeAdminAccess;
  (window as any).checkAdmin = checkAdminAccess;
  
  console.log('🔧 Admin utilities loaded! Use:');
  console.log('  - grantAdmin() to grant access');
  console.log('  - revokeAdmin() to revoke access');
  console.log('  - checkAdmin() to check status');
} 