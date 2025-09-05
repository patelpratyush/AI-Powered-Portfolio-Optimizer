import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LoadingSpinner, FullScreenLoader, ButtonSpinner } from '@/components/LoadingSpinner';

describe('LoadingSpinner', () => {
  it('renders with default props', () => {
    render(<LoadingSpinner />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  it('renders with custom text', () => {
    render(<LoadingSpinner text="Custom loading message" />);
    
    expect(screen.getByText('Custom loading message')).toBeInTheDocument();
  });

  it('renders without text when text prop is empty', () => {
    render(<LoadingSpinner text="" />);
    
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<LoadingSpinner className="custom-class" />);
    
    const spinner = screen.getByTestId('loading-spinner');
    expect(spinner).toHaveClass('custom-class');
  });

  it('renders different sizes correctly', () => {
    const { rerender } = render(<LoadingSpinner size="sm" />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

    rerender(<LoadingSpinner size="lg" />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
});

describe('FullScreenLoader', () => {
  it('renders with overlay', () => {
    render(<FullScreenLoader />);
    
    const overlay = screen.getByTestId('fullscreen-loader');
    expect(overlay).toBeInTheDocument();
    expect(overlay).toHaveClass('fixed inset-0');
  });

  it('renders with custom message', () => {
    render(<FullScreenLoader message="Processing your request..." />);
    
    expect(screen.getByText('Processing your request...')).toBeInTheDocument();
  });

  it('renders with progress when provided', () => {
    render(<FullScreenLoader progress={75} />);
    
    expect(screen.getByText('75%')).toBeInTheDocument();
  });
});

describe('ButtonSpinner', () => {
  it('renders small spinner for buttons', () => {
    render(<ButtonSpinner />);
    
    const spinner = screen.getByTestId('button-spinner');
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass('animate-spin');
  });

  it('applies custom className', () => {
    render(<ButtonSpinner className="custom-button-spinner" />);
    
    const spinner = screen.getByTestId('button-spinner');
    expect(spinner).toHaveClass('custom-button-spinner');
  });
});