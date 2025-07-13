import * as React from "react"

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  children: React.ReactNode
}

interface SelectOptionProps extends React.OptionHTMLAttributes<HTMLOptionElement> {
  children: React.ReactNode
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", children, ...props }, ref) => (
    <select
      ref={ref}
      className={`block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm bg-white ${className}`}
      {...props}
    >
      {children}
    </select>
  )
)

Select.displayName = "Select"

const SelectOption: React.FC<SelectOptionProps> = ({ children, ...props }) => (
  <option {...props}>
    {children}
  </option>
)

export { Select, SelectOption } 