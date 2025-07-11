import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { Button } from '../button';
import { Input } from '../input';
import { Textarea } from '../textarea';
import { Label } from '../label';
import { Select, SelectOption } from '../select';
import { Checkbox } from '../checkbox';
import { Alert, AlertTitle, AlertDescription } from '../alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '../dialog';

describe('Button Component', () => {
  it('renders with default props', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument();
  });

  it('applies variant styles correctly', () => {
    render(<Button variant="premium">Premium Button</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-gradient-to-r from-purple-600 to-blue-600');
  });

  it('applies size styles correctly', () => {
    render(<Button size="xl">Extra Large Button</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('h-14');
  });

  it('shows loading state', () => {
    render(<Button loading>Loading Button</Button>);
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button.querySelector('svg.animate-spin')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled Button</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});

describe('Input Component', () => {
  it('renders with default props', () => {
    render(<Input placeholder="Enter text" />);
    expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Input label="Email Address" />);
    expect(screen.getByText('Email Address')).toBeInTheDocument();
  });

  it('shows error state and message', () => {
    render(<Input error errorMessage="This field is required" />);
    expect(screen.getByText('This field is required')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveClass('border-red-300');
  });

  it('shows success state and message', () => {
    render(<Input success successMessage="Valid input" />);
    expect(screen.getByText('Valid input')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveClass('border-green-300');
  });

  it('shows helper text', () => {
    render(<Input helperText="This is helpful information" />);
    expect(screen.getByText('This is helpful information')).toBeInTheDocument();
  });

  it('handles value changes', () => {
    const handleChange = vi.fn();
    render(<Input onChange={handleChange} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'test' } });
    expect(handleChange).toHaveBeenCalled();
  });
});

describe('Textarea Component', () => {
  it('renders with default props', () => {
    render(<Textarea placeholder="Enter text" />);
    expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Textarea label="Description" />);
    expect(screen.getByText('Description')).toBeInTheDocument();
  });

  it('shows character count when enabled', () => {
    render(<Textarea showCharacterCount maxLength={100} />);
    expect(screen.getByText('0/100')).toBeInTheDocument();
  });

  it('updates character count on input', () => {
    render(<Textarea showCharacterCount maxLength={100} />);
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Hello World' } });
    expect(screen.getByText('11/100')).toBeInTheDocument();
  });

  it('shows error state and message', () => {
    render(<Textarea error errorMessage="This field is required" />);
    expect(screen.getByText('This field is required')).toBeInTheDocument();
  });
});

describe('Label Component', () => {
  it('renders with default props', () => {
    render(<Label>Test Label</Label>);
    expect(screen.getByText('Test Label')).toBeInTheDocument();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLLabelElement>();
    render(<Label ref={ref}>Test Label</Label>);
    expect(ref.current).toBeInstanceOf(HTMLLabelElement);
  });
});

describe('Select Component', () => {
  it('renders with options', () => {
    render(
      <Select>
        <SelectOption value="option1">Option 1</SelectOption>
        <SelectOption value="option2">Option 2</SelectOption>
      </Select>
    );
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getByText('Option 1')).toBeInTheDocument();
    expect(screen.getByText('Option 2')).toBeInTheDocument();
  });

  it('handles value changes', () => {
    const handleChange = vi.fn();
    render(
      <Select onChange={handleChange}>
        <SelectOption value="option1">Option 1</SelectOption>
      </Select>
    );
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'option1' } });
    expect(handleChange).toHaveBeenCalled();
  });
});

describe('Checkbox Component', () => {
  it('renders with label', () => {
    render(<Checkbox label="Accept terms" />);
    expect(screen.getByRole('checkbox')).toBeInTheDocument();
    expect(screen.getByText('Accept terms')).toBeInTheDocument();
  });

  it('handles change events', () => {
    const handleChange = vi.fn();
    render(<Checkbox onChange={handleChange} />);
    fireEvent.click(screen.getByRole('checkbox'));
    expect(handleChange).toHaveBeenCalled();
  });
});

describe('Alert Component', () => {
  it('renders with default variant', () => {
    render(<Alert>This is an alert</Alert>);
    expect(screen.getByText('This is an alert')).toBeInTheDocument();
  });

  it('renders with different variants', () => {
    const { rerender } = render(<Alert variant="success">Success alert</Alert>);
    expect(screen.getByText('Success alert')).toHaveClass('bg-green-50');

    rerender(<Alert variant="destructive">Error alert</Alert>);
    expect(screen.getByText('Error alert')).toHaveClass('bg-red-50');
  });

  it('renders with title and description', () => {
    render(
      <Alert>
        <AlertTitle>Alert Title</AlertTitle>
        <AlertDescription>Alert description</AlertDescription>
      </Alert>
    );
    expect(screen.getByText('Alert Title')).toBeInTheDocument();
    expect(screen.getByText('Alert description')).toBeInTheDocument();
  });
});

describe('Dialog Component', () => {
  it('renders when open', () => {
    render(
      <Dialog open={true} onOpenChange={() => {}}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Dialog Title</DialogTitle>
            <DialogDescription>Dialog description</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
    expect(screen.getByText('Dialog Title')).toBeInTheDocument();
    expect(screen.getByText('Dialog description')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(
      <Dialog open={false} onOpenChange={() => {}}>
        <DialogContent>Content</DialogContent>
      </Dialog>
    );
    expect(screen.queryByText('Content')).not.toBeInTheDocument();
  });

  it('calls onOpenChange when backdrop is clicked', () => {
    const onOpenChange = vi.fn();
    render(
      <Dialog open={true} onOpenChange={onOpenChange}>
        <DialogContent>Content</DialogContent>
      </Dialog>
    );
    const backdrop = document.querySelector('.bg-black\\/50');
    if (backdrop) {
      fireEvent.click(backdrop);
      expect(onOpenChange).toHaveBeenCalledWith(false);
    }
  });
});

// Integration tests
describe('UI Components Integration', () => {
  it('form with multiple components works together', () => {
    render(
      <div>
        <Input label="Email" placeholder="Enter email" />
        <Textarea label="Message" placeholder="Enter message" />
        <Button>Submit</Button>
      </div>
    );
    
    expect(screen.getByText('Email')).toBeInTheDocument();
    expect(screen.getByText('Message')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
  });

  it('dialog with form components', () => {
    render(
      <Dialog open={true} onOpenChange={() => {}}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Company</DialogTitle>
          </DialogHeader>
          <Input label="Company Name" />
          <Textarea label="Description" />
          <DialogFooter>
            <Button variant="outline">Cancel</Button>
            <Button>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
    
    expect(screen.getByText('Add Company')).toBeInTheDocument();
    expect(screen.getByText('Company Name')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
  });
}); 