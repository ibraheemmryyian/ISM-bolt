import React, { useState } from 'react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

const screeningQuestions = [
  {
    id: 'motivation',
    label: 'Why do you want to join SymbioFlows?',
    type: 'textarea',
    required: true,
  },
  {
    id: 'industry',
    label: 'What industry are you in?',
    type: 'select',
    options: ['Manufacturing', 'Food & Beverage', 'Textiles', 'Chemicals', 'Electronics', 'Healthcare', 'Construction', 'Other'],
    required: true,
  },
  {
    id: 'waste_streams',
    label: 'What types of waste or byproducts does your company generate?',
    type: 'textarea',
    required: true,
  },
  {
    id: 'sustainability_goals',
    label: 'What are your main sustainability goals?',
    type: 'textarea',
    required: true,
  }
];

interface FormState {
  company_name: string;
  contact_email: string;
  contact_name: string;
  answers: Record<string, string>;
}

export default function RequestAccess() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>({
    company_name: '',
    contact_email: '',
    contact_name: '',
    answers: screeningQuestions.reduce((acc, q) => ({ ...acc, [q.id]: '' }), {} as Record<string, string>),
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [creatingAccount, setCreatingAccount] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    if (screeningQuestions.some(q => q.id === name)) {
      setForm(f => ({ ...f, answers: { ...f.answers, [name]: value } }));
    } else {
      setForm(f => ({ ...f, [name]: value }));
    }
  };

  const validate = () => {
    if (!form.company_name.trim()) return 'Company name is required.';
    if (!form.contact_name.trim()) return 'Contact name is required.';
    if (!form.contact_email.trim() || !form.contact_email.includes('@')) return 'Valid contact email is required.';
    for (const q of screeningQuestions) {
      if (q.required && !form.answers[q.id].trim()) return `Please answer: ${q.label}`;
    }
    return null;
  };

  const createAccountAndSignIn = async () => {
    setCreatingAccount(true);
    try {
      // Create account with email
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email: form.contact_email,
        password: `SymbioFlows${Date.now()}`, // Temporary password
        options: {
          data: {
            company_name: form.company_name,
            contact_name: form.contact_name,
            role: 'company_admin'
          }
        }
      });

      if (authError) throw authError;

      // Create company profile
      const { data: companyData, error: companyError } = await supabase
        .from('companies')
        .insert({
          name: form.company_name,
          industry: form.answers.industry || 'Unknown',
          location: 'To be updated', // Will be updated during onboarding
          employee_count: 0, // Will be updated during onboarding
          contact_email: form.contact_email,
          contact_name: form.contact_name,
          onboarding_completed: false,
          application_answers: form.answers
        })
        .select()
        .single();

      if (companyError) throw companyError;

      // Store company ID in session
      sessionStorage.setItem('current_company_id', companyData.id);

      // Navigate to AI onboarding
      navigate('/onboarding', { 
        state: { 
          companyId: companyData.id,
          autoCreated: true 
        } 
      });

    } catch (err: any) {
      console.error('Account creation error:', err);
      setError('Failed to create account. Please try signing in manually.');
      setCreatingAccount(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const validation = validate();
    if (validation) {
      setError(validation);
      return;
    }
    setSubmitting(true);
    try {
      console.log('Submitting application:', {
        company_name: form.company_name,
        contact_name: form.contact_name,
        contact_email: form.contact_email,
        application_answers: form.answers,
      });
      
      const { data, error: supaError } = await supabase.from('company_applications').insert({
        company_name: form.company_name,
        contact_name: form.contact_name,
        contact_email: form.contact_email,
        application_answers: form.answers,
        status: 'pending'
      }).select();
      
      console.log('Application submission result:', { data, error: supaError });
      
      if (supaError) throw supaError;
      
      console.log('Application submitted successfully:', data);
      setSuccess(true);
      
      // Automatically create account and sign in
      setTimeout(() => {
        createAccountAndSignIn();
      }, 2000);
      
    } catch (err: any) {
      console.error('Application submission error:', err);
      setError(err.message || 'Submission failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-emerald-50 via-white to-blue-50">
        <div className="bg-white rounded-xl shadow p-8 max-w-lg w-full text-center">
          <div className="animate-pulse">
            <h2 className="text-2xl font-bold mb-2 text-emerald-700">Application Approved!</h2>
            <p className="text-slate-600 mb-4">Creating your account and setting up your workspace...</p>
            {creatingAccount && (
              <div className="flex items-center justify-center space-x-2">
                <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-emerald-600">Setting up your account...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-emerald-50 via-white to-blue-50">
      <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow p-8 max-w-lg w-full space-y-6">
        <h2 className="text-2xl font-bold text-emerald-700 mb-2">Request Access to SymbioFlows</h2>
        <p className="text-slate-600 mb-4">For industrial companies only. We'll create your account automatically upon approval.</p>
        {error && (
          <div className="bg-red-100 text-red-700 p-2 rounded mb-4 text-center">{error}</div>
        )}
        {loading && (
          <Loader2 className="h-6 w-6 animate-spin mx-auto my-4" />
        )}
        
        <div className="space-y-2">
          <label className="block text-slate-700 font-medium">Company Name *</label>
          <input 
            name="company_name" 
            value={form.company_name} 
            onChange={handleChange} 
            className="w-full border border-slate-200 rounded p-2" 
            required 
          />
        </div>
        
        <div className="space-y-2">
          <label className="block text-slate-700 font-medium">Contact Name *</label>
          <input 
            name="contact_name" 
            value={form.contact_name} 
            onChange={handleChange} 
            className="w-full border border-slate-200 rounded p-2" 
            required 
          />
        </div>
        
        <div className="space-y-2">
          <label className="block text-slate-700 font-medium">Contact Email *</label>
          <input 
            name="contact_email" 
            type="email" 
            value={form.contact_email} 
            onChange={handleChange} 
            className="w-full border border-slate-200 rounded p-2" 
            required 
          />
        </div>
        
        {screeningQuestions.map(q => (
          <div className="space-y-2" key={q.id}>
            <label className="block text-slate-700 font-medium">{q.label} *</label>
            {q.type === 'textarea' ? (
              <textarea 
                name={q.id} 
                value={form.answers[q.id]} 
                onChange={handleChange} 
                className="w-full border border-slate-200 rounded p-2" 
                rows={3} 
                required={q.required} 
              />
            ) : q.type === 'select' ? (
              <select 
                name={q.id} 
                value={form.answers[q.id]} 
                onChange={handleChange} 
                className="w-full border border-slate-200 rounded p-2" 
                required={q.required}
              >
                <option value="">Select an option</option>
                {q.options?.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            ) : (
              <input 
                name={q.id} 
                value={form.answers[q.id]} 
                onChange={handleChange} 
                className="w-full border border-slate-200 rounded p-2" 
                required={q.required} 
              />
            )}
          </div>
        ))}
        
        <button 
          type="submit" 
          className="w-full bg-gradient-to-r from-emerald-500 to-blue-500 text-white font-bold py-2 rounded mt-4 disabled:opacity-60" 
          disabled={submitting || loading}
        >
          {submitting ? 'Submitting...' : 'Request Access & Create Account'}
        </button>
      </form>
    </div>
  );
} 