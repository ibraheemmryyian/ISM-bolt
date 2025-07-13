import * as React from "react"

interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className = "", label, ...props }, ref) => (
    <div className="flex items-center">
      <input
        type="checkbox"
        ref={ref}
        className={`h-4 w-4 text-emerald-600 focus:ring-emerald-500 border-gray-300 rounded ${className}`}
        {...props}
      />
      {label && (
        <label className="ml-2 block text-sm text-gray-900">
          {label}
        </label>
      )}
    </div>
  )
)

Checkbox.displayName = "Checkbox"

export { Checkbox } 