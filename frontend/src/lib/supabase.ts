import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    'Missing Supabase environment variables. Please check your .env file and ensure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are set and not quoted.'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function isUserAdmin(userId: string): Promise<boolean> {
  const { data, error } = await supabase
    .from('companies')
    .select('role')
    .eq('id', userId)
    .single();
  if (error) {
    console.error('Error checking admin status:', error);
    return false;
  }
  return data?.role === 'admin';
}