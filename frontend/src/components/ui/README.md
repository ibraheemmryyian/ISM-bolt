# ISM UI Component Library

A comprehensive, enterprise-grade UI component library built with React, TypeScript, and Tailwind CSS, designed for the Industrial Symbiosis Marketplace platform.

## 🎯 Design Philosophy

- **Monopoly Edge**: Every component provides competitive advantage through superior UX
- **Accessibility First**: WCAG 2.1 AA compliant components
- **Performance Optimized**: Minimal bundle size with maximum functionality
- **Enterprise Ready**: Built for Fortune 500 adoption

## 🚀 Components

### Button
Advanced button component with multiple variants, loading states, and icons.

```tsx
import { Button } from './ui/button';

// Basic usage
<Button>Click me</Button>

// With variants
<Button variant="premium" size="lg" loading>
  Premium Action
</Button>

// With icons
<Button icon={<Icon />} iconPosition="right">
  With Icon
</Button>
```

**Variants:** `default`, `outline`, `ghost`, `destructive`, `secondary`, `premium`, `gradient`
**Sizes:** `sm`, `md`, `lg`, `xl`

### Input
Enhanced input component with validation states, icons, and accessibility features.

```tsx
import { Input } from './ui/input';

// Basic usage
<Input placeholder="Enter text" />

// With validation
<Input 
  label="Email"
  error={hasError}
  errorMessage="Invalid email"
  success={isValid}
  successMessage="Valid email"
  helperText="We'll never share your email"
/>

// With icons
<Input 
  icon={<MailIcon />}
  iconPosition="left"
  label="Email Address"
/>
```

### Textarea
Advanced textarea with character counting, auto-resize, and validation.

```tsx
import { Textarea } from './ui/textarea';

// Basic usage
<Textarea placeholder="Enter description" />

// With character count
<Textarea 
  maxLength={500}
  showCharacterCount
  label="Description"
/>

// Auto-resize
<Textarea 
  autoResize
  label="Dynamic Height"
/>
```

### Card
Flexible card component for content containers.

```tsx
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
  </CardHeader>
  <CardContent>
    Card content goes here
  </CardContent>
</Card>
```

### Dialog
Modal dialog component with backdrop and accessibility features.

```tsx
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';

<Dialog open={isOpen} onOpenChange={setIsOpen}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
      <DialogDescription>Dialog description</DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline">Cancel</Button>
      <Button>Confirm</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

### Select
Dropdown select component with options.

```tsx
import { Select, SelectOption } from './ui/select';

<Select onChange={handleChange}>
  <SelectOption value="option1">Option 1</SelectOption>
  <SelectOption value="option2">Option 2</SelectOption>
</Select>
```

### Checkbox
Accessible checkbox component with label support.

```tsx
import { Checkbox } from './ui/checkbox';

<Checkbox label="Accept terms and conditions" />
```

### Alert
Notification component with multiple variants.

```tsx
import { Alert, AlertTitle, AlertDescription } from './ui/alert';

<Alert variant="success">
  <AlertTitle>Success!</AlertTitle>
  <AlertDescription>Operation completed successfully.</AlertDescription>
</Alert>
```

**Variants:** `default`, `destructive`, `success`, `warning`

### Label
Accessible label component for form elements.

```tsx
import { Label } from './ui/label';

<Label htmlFor="input-id">Input Label</Label>
```

### Badge
Small status indicator component.

```tsx
import { Badge } from './ui/badge';

<Badge>New</Badge>
```

### Progress
Progress indicator component.

```tsx
import { Progress } from './ui/progress';

<Progress value={75} />
```

## 🧪 Testing

All components include comprehensive unit tests using Jest and React Testing Library.

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run specific component tests
npm test -- ui-components.test.tsx
```

## 🎨 Customization

### Theme Colors
The component library uses a consistent color palette based on Tailwind CSS:

- **Primary:** Emerald (green) - represents sustainability and growth
- **Secondary:** Gray - neutral tones for secondary actions
- **Success:** Green - positive states and confirmations
- **Warning:** Yellow - caution and warnings
- **Error:** Red - errors and destructive actions
- **Premium:** Purple/Blue gradient - premium features

### Custom Variants
To add custom variants, extend the component interfaces:

```tsx
// Extend ButtonProps
type CustomButtonProps = ButtonProps & {
  variant?: 'custom' | ButtonProps['variant'];
};

// Add custom styles
const customVariantStyles = {
  custom: 'bg-custom-color text-white hover:bg-custom-color-dark',
  ...variantStyles
};
```

## ♿ Accessibility

All components follow WCAG 2.1 AA guidelines:

- **Keyboard Navigation:** All interactive elements are keyboard accessible
- **Screen Readers:** Proper ARIA labels and roles
- **Focus Management:** Visible focus indicators
- **Color Contrast:** Meets AA contrast requirements
- **Semantic HTML:** Proper HTML structure and semantics

## 📱 Responsive Design

Components are built with mobile-first responsive design:

- **Breakpoints:** Follow Tailwind CSS breakpoint system
- **Touch Targets:** Minimum 44px touch targets for mobile
- **Flexible Layouts:** Components adapt to different screen sizes
- **Performance:** Optimized for mobile performance

## 🔧 Development

### Adding New Components

1. Create component file in `frontend/src/components/ui/`
2. Follow naming convention: `component-name.tsx`
3. Export component with proper TypeScript types
4. Add comprehensive unit tests
5. Update this README with documentation

### Component Structure

```tsx
import React from 'react';

interface ComponentProps {
  // Define props with TypeScript
}

export const Component: React.FC<ComponentProps> = ({ 
  // Destructure props
}) => {
  // Component logic
  
  return (
    // JSX with Tailwind classes
  );
};

Component.displayName = 'Component';
```

### Best Practices

- **TypeScript First:** All components use TypeScript
- **Forward Refs:** Use `React.forwardRef` for form components
- **Display Names:** Set `displayName` for debugging
- **Consistent Styling:** Use Tailwind CSS classes
- **Error Boundaries:** Handle errors gracefully
- **Performance:** Optimize re-renders with `React.memo` when needed

## 🚀 Performance

- **Bundle Size:** Minimal impact on bundle size
- **Tree Shaking:** Components can be imported individually
- **Lazy Loading:** Support for code splitting
- **Memoization:** Optimized re-renders
- **CSS-in-JS:** No runtime CSS-in-JS overhead

## 📦 Installation

The UI components are part of the ISM platform and don't require separate installation. They're available at:

```
frontend/src/components/ui/
```

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure accessibility compliance
5. Test across different browsers and devices

## 📄 License

Part of the ISM platform - proprietary software. 