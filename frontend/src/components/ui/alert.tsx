import * as React from "react"

interface AlertProps {
  variant?: 'default' | 'destructive' | 'success' | 'warning'
  children: React.ReactNode
  className?: string
}

const Alert: React.FC<AlertProps> = ({ 
  variant = 'default', 
  children, 
  className = "" 
}) => {
  const baseStyles = "p-4 rounded-md border"
  
  const variantStyles = {
    default: "bg-blue-50 border-blue-200 text-blue-800",
    destructive: "bg-red-50 border-red-200 text-red-800",
    success: "bg-green-50 border-green-200 text-green-800",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-800"
  }

  return (
    <div className={`${baseStyles} ${variantStyles[variant]} ${className}`}>
      {children}
    </div>
  )
}

interface AlertTitleProps {
  children: React.ReactNode
  className?: string
}

const AlertTitle: React.FC<AlertTitleProps> = ({ children, className = "" }) => (
  <h5 className={`font-medium mb-1 ${className}`}>
    {children}
  </h5>
)

interface AlertDescriptionProps {
  children: React.ReactNode
  className?: string
}

const AlertDescription: React.FC<AlertDescriptionProps> = ({ children, className = "" }) => (
  <div className={`text-sm ${className}`}>
    {children}
  </div>
)

export { Alert, AlertTitle, AlertDescription } 