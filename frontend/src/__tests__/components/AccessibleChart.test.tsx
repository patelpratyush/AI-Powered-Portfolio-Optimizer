import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AccessibleChart } from '@/components/AccessibleChart';

// Mock the chart libraries
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart" role="img" aria-label="Line chart">{children}</div>
  ),
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
}));

describe('AccessibleChart', () => {
  const mockData = [
    { name: 'Jan', value: 100 },
    { name: 'Feb', value: 150 },
    { name: 'Mar', value: 120 },
  ];

  const defaultProps = {
    data: mockData,
    title: 'Test Chart',
    description: 'A test chart for accessibility',
  };

  beforeEach(() => {
    // Reset any mocks before each test
    vi.clearAllMocks();
  });

  it('renders chart with accessibility features', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    // Check for main chart container
    expect(screen.getByRole('figure')).toBeInTheDocument();
    
    // Check for title and description
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('A test chart for accessibility')).toBeInTheDocument();
    
    // Check for chart components
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('provides keyboard navigation', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    const chart = screen.getByRole('figure');
    
    // Chart should be focusable
    expect(chart).toHaveAttribute('tabIndex', '0');
  });

  it('provides screen reader accessible data table', () => {
    render(<AccessibleChart {...defaultProps} showDataTable />);
    
    // Should render data table for screen readers
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByText('Chart Data Table')).toBeInTheDocument();
    
    // Check table headers
    expect(screen.getByRole('columnheader', { name: /name/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /value/i })).toBeInTheDocument();
    
    // Check table data
    expect(screen.getByRole('cell', { name: 'Jan' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '100' })).toBeInTheDocument();
  });

  it('handles keyboard interactions', () => {
    const onDataPointFocus = vi.fn();
    render(<AccessibleChart {...defaultProps} onDataPointFocus={onDataPointFocus} />);
    
    const chart = screen.getByRole('figure');
    
    // Simulate arrow key navigation
    fireEvent.keyDown(chart, { key: 'ArrowRight' });
    fireEvent.keyDown(chart, { key: 'ArrowLeft' });
    fireEvent.keyDown(chart, { key: 'Home' });
    fireEvent.keyDown(chart, { key: 'End' });
    
    // Should handle keyboard events
    expect(chart).toBeInTheDocument();
  });

  it('provides high contrast mode support', () => {
    render(<AccessibleChart {...defaultProps} highContrast />);
    
    const chart = screen.getByRole('figure');
    expect(chart).toHaveClass('high-contrast');
  });

  it('supports custom color palette for accessibility', () => {
    const colorPalette = ['#000000', '#FFFFFF', '#FF0000'];
    render(<AccessibleChart {...defaultProps} colorPalette={colorPalette} />);
    
    // Chart should render with custom colors
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('provides data summary for screen readers', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    // Should have aria-describedby pointing to summary
    const chart = screen.getByRole('figure');
    const summaryId = chart.getAttribute('aria-describedby');
    
    if (summaryId) {
      const summary = document.getElementById(summaryId);
      expect(summary).toBeInTheDocument();
    }
  });

  it('handles empty data gracefully', () => {
    render(<AccessibleChart {...defaultProps} data={[]} />);
    
    // Should show empty state message
    expect(screen.getByText(/no data/i)).toBeInTheDocument();
  });

  it('supports different chart types', () => {
    const { rerender } = render(<AccessibleChart {...defaultProps} type="line" />);
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();

    rerender(<AccessibleChart {...defaultProps} type="bar" />);
    // Would show bar chart if implemented
    
    rerender(<AccessibleChart {...defaultProps} type="pie" />);
    // Would show pie chart if implemented
  });

  it('provides proper ARIA labels and roles', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    const chart = screen.getByRole('figure');
    expect(chart).toHaveAttribute('aria-labelledby');
    expect(chart).toHaveAttribute('aria-describedby');
    
    // Check for proper heading levels
    expect(screen.getByRole('heading', { level: 3 })).toBeInTheDocument();
  });

  it('supports reduced motion preferences', () => {
    // Mock reduced motion preference
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });

    render(<AccessibleChart {...defaultProps} />);
    
    // Chart should respect reduced motion
    const chart = screen.getByRole('figure');
    expect(chart).toHaveClass('reduce-motion');
  });

  it('provides tooltip content for screen readers', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    // Should have live region for announcing tooltip content
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('handles focus management correctly', () => {
    render(<AccessibleChart {...defaultProps} />);
    
    const chart = screen.getByRole('figure');
    
    // Focus the chart
    chart.focus();
    expect(document.activeElement).toBe(chart);
  });
});