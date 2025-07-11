import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

interface ValidationRule {
  test: (value: string) => boolean;
  message: string;
}

interface FormValidationProps {
  value: string;
  rules: ValidationRule[];
  onValidationChange?: (isValid: boolean) => void;
  showValidation?: boolean;
}

export function FormValidation({ 
  value, 
  rules, 
  onValidationChange, 
  showValidation = true 
}: FormValidationProps) {
  const [errors, setErrors] = useState<string[]>([]);
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    if (!showValidation) {
      setErrors([]);
      setIsValid(true);
      onValidationChange?.(true);
      return;
    }

    const newErrors: string[] = [];
    
    rules.forEach(rule => {
      if (!rule.test(value)) {
        newErrors.push(rule.message);
      }
    });

    setErrors(newErrors);
    const valid = newErrors.length === 0;
    setIsValid(valid);
    onValidationChange?.(valid);
  }, [value, rules, showValidation, onValidationChange]);

  if (!showValidation) return null;

  return (
    <div className="mt-1">
      {errors.length > 0 ? (
        <div className="flex items-center text-red-600 text-sm">
          <AlertCircle className="h-4 w-4 mr-1" />
          <span>{errors[0]}</span>
        </div>
      ) : value.length > 0 ? (
        <div className="flex items-center text-green-600 text-sm">
          <CheckCircle className="h-4 w-4 mr-1" />
          <span>Valid</span>
        </div>
      ) : null}
    </div>
  );
}

// Common validation rules
export const validationRules = {
  required: (fieldName: string): ValidationRule => ({
    test: (value: string) => value.trim().length > 0,
    message: `${fieldName} is required`
  }),
  
  email: (): ValidationRule => ({
    test: (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
    message: 'Please enter a valid email address'
  }),
  
  minLength: (length: number): ValidationRule => ({
    test: (value: string) => value.length >= length,
    message: `Must be at least ${length} characters`
  }),
  
  maxLength: (length: number): ValidationRule => ({
    test: (value: string) => value.length <= length,
    message: `Must be no more than ${length} characters`
  }),
  
  username: (): ValidationRule => ({
    test: (value: string) => /^[a-zA-Z0-9_]{3,20}$/.test(value),
    message: 'Username must be 3-20 characters, letters, numbers, and underscores only'
  }),
  
  password: (): ValidationRule => ({
    test: (value: string) => /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/.test(value),
    message: 'Password must be at least 8 characters with uppercase, lowercase, and number'
  }),
  
  companyName: (): ValidationRule => ({
    test: (value: string) => value.trim().length >= 2 && value.trim().length <= 100,
    message: 'Company name must be between 2 and 100 characters'
  }),
  
  industry: (): ValidationRule => ({
    test: (value: string) => value.trim().length >= 2,
    message: 'Please select or enter an industry'
  })
}; 