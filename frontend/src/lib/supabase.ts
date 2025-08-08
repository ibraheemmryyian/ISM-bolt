import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  },
  global: {
    headers: {
      'X-Client-Info': 'symbioflows-web'
    }
  }
});

// API base URL for backend calls
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://api.symbioflows.com'
export const AI_SERVICES_URL = import.meta.env.VITE_AI_SERVICES_URL || 'https://api.symbioflows.com/ai'

// Test Supabase connection
export async function testSupabaseConnection() {
  try {
    const { data, error } = await supabase
      .from('companies')
      .select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('Supabase connection test failed:', error);
      return false;
    }
    
    console.log('Supabase connection test successful');
    return true;
  } catch (error) {
    console.error('Supabase connection test error:', error);
    return false;
  }
}

// Add error handling middleware
supabase.auth.onAuthStateChange((event, session) => {
  if (event === 'SIGNED_OUT') {
    // Clear any cached data
    localStorage.removeItem('user_preferences');
    localStorage.removeItem('onboarding_complete');
  }
});

// Retry logic for failed requests
export const withRetry = async (fn: () => Promise<any>, maxRetries = 3) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      if (i === maxRetries - 1) throw error;
      if (error?.code === 'PGRST301' || error?.code === 'PGRST302') {
        // Network error, retry after delay
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        continue;
      }
      throw error;
    }
  }
};

export default supabase;

// Admin status check
export async function isUserAdmin(userId: string): Promise<boolean> {
  try {
    // Check companies table for admin role
    const { data: companyData, error: companyError } = await supabase
      .from('companies')
      .select('role')
      .eq('id', userId)
      .maybeSingle();
    
    if (companyError) {
      console.error('Error checking admin status:', companyError);
      return false;
    }
    
    return companyData?.role === 'admin';
  } catch (error) {
    console.error('Error checking admin status:', error);
    return false;
  }
}

// Get user by email
export async function getUserByEmail(email: string) {
  try {
    const { data, error } = await supabase
      .from('companies')
      .select('*')
      .eq('email', email)
      .maybeSingle();
    
    if (error) {
      throw error;
    }
    
    return data;
  } catch (error) {
    throw error;
  }
}

// Sign in with username (looks up email)
export async function signInWithUsername(username: string, password: string) {
  try {
    // First get the user's email by username
    const userProfile = await getUserByUsername(username);
    
    if (!userProfile) {
      throw new Error('Invalid username or password');
    }
    
    // Sign in with the email
    const { data, error } = await supabase.auth.signInWithPassword({
      email: userProfile.email,
      password: password,
    });
    
    if (error) {
      // Handle specific error cases
      if (error.message.includes('Email not confirmed')) {
        throw new Error('Please verify your email address before signing in. Check your inbox for a verification link.');
      }
      if (error.message.includes('Invalid login credentials')) {
        throw new Error('Invalid username or password');
      }
      throw error;
    }
    
    return data;
  } catch (error) {
    throw error;
  }
}

// Sign up with real email and verification
export async function signUpWithEmail(password: string, companyName: string, email: string) {
  try {
    // Check if email already exists
    const { data: existingEmail } = await supabase
      .from('companies')
      .select('email')
      .eq('email', email)
      .maybeSingle();

    if (existingEmail) {
      throw new Error('Email already registered. Please use a different email or sign in.');
    }

    // Create auth user with real email (Supabase will send verification email)
    const { data, error } = await supabase.auth.signUp({
      email: email,
      password: password,
      options: {
        data: {
          company_name: companyName,
        },
        emailRedirectTo: `${window.location.origin}/auth/callback`
      }
    });
    
    if (error) {
      throw error;
    }
    
    return data;
  } catch (error) {
    throw error;
  }
}

// Check if user email is verified
export async function isEmailVerified(): Promise<boolean> {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    return user?.email_confirmed_at !== null;
  } catch (error) {
    console.error('Error checking email verification:', error);
    return false;
  }
}

// Resend verification email
export async function resendVerificationEmail(email: string) {
  try {
    const { error } = await supabase.auth.resend({
      type: 'signup',
      email: email,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback`
      }
    });
    
    if (error) {
      throw error;
    }
  } catch (error) {
    throw error;
  }
}

// Send password reset email
export async function sendPasswordResetEmail(email: string) {
  try {
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/reset-password`
    });
    
    if (error) {
      throw error;
    }
  } catch (error) {
    throw error;
  }
}

// Update user password
export async function updateUserPassword(newPassword: string) {
  try {
    const { error } = await supabase.auth.updateUser({
      password: newPassword,
    });
    
    if (error) {
      throw error;
    }
  } catch (error) {
    throw error;
  }
}

// Get current user session
export async function getCurrentSession() {
  try {
    const { data: { session }, error } = await supabase.auth.getSession();
    
    if (error) {
      throw error;
    }
    
    return session;
  } catch (error) {
    console.error('Error getting current session:', error);
    return null;
  }
}

// Sign out user
export async function signOut() {
  try {
    const { error } = await supabase.auth.signOut();
    
    if (error) {
      throw error;
    }
  } catch (error) {
    throw error;
  }
}