import React, { useRef, useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AriaUtils, KeyboardNavigation, ContentAccessibility } from '../utils/accessibility';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface DataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface AccessibleChartProps {
  data: DataPoint[];
  title: string;
  description?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  className?: string;
}

export const AccessibleChart: React.FC<AccessibleChartProps> = ({
  data,
  title,
  description,
  xAxisLabel = 'X Axis',
  yAxisLabel = 'Y Axis',
  className = ''
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const [isTableView, setIsTableView] = useState(false);
  const [currentDataIndex, setCurrentDataIndex] = useState(0);
  const [announcementId] = useState(() => AriaUtils.generateId('chart-announcement'));
  const [tableId] = useState(() => AriaUtils.generateId('chart-table'));
  const [descriptionId] = useState(() => AriaUtils.generateId('chart-desc'));

  useEffect(() => {
    const currentRef = chartRef.current;
    if (currentRef) {
      const handleKeyDown = (e: KeyboardEvent) => {
        if (!isTableView) return;
        
        const newIndex = KeyboardNavigation.handleArrowKeys(e, [], currentDataIndex);
        if (newIndex !== null && newIndex < data.length) {
          setCurrentDataIndex(newIndex);
          const point = data[newIndex];
          AriaUtils.announce(
            `${point.name}: ${point.value}. Point ${newIndex + 1} of ${data.length}`,
            'polite'
          );
        }
      };

      currentRef.addEventListener('keydown', handleKeyDown);
      return () => currentRef.removeEventListener('keydown', handleKeyDown);
    }
  }, [isTableView, currentDataIndex, data]);

  const toggleView = () => {
    setIsTableView(!isTableView);
    AriaUtils.announce(
      isTableView ? 'Switched to chart view' : 'Switched to table view',
      'polite'
    );
  };

  const chartDescription = ContentAccessibility.generateDescription({
    type: 'chart',
    label: description || title
  });

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle id={descriptionId}>{title}</CardTitle>
            {description && (
              <CardDescription>
                {description}
              </CardDescription>
            )}
          </div>
          <Button
            onClick={toggleView}
            variant="outline"
            size="sm"
            aria-label={`Switch to ${isTableView ? 'chart' : 'table'} view`}
          >
            {isTableView ? 'Show Chart' : 'Show Table'}
          </Button>
        </div>
      </CardHeader>
      
      <CardContent>
        <div
          ref={chartRef}
          role={isTableView ? 'application' : 'img'}
          aria-label={chartDescription}
          aria-describedby={descriptionId}
          tabIndex={isTableView ? 0 : -1}
        >
          {!isTableView ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data} aria-hidden="true">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  aria-label={xAxisLabel}
                />
                <YAxis aria-label={yAxisLabel} />
                <Tooltip 
                  formatter={(value, name) => [value, name]}
                  labelFormatter={(label) => `${xAxisLabel}: ${label}`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#2563eb" 
                  strokeWidth={2}
                  dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: '#2563eb', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="overflow-x-auto">
              <table 
                id={tableId}
                className="w-full border-collapse border border-border"
                role="table"
                aria-label={`Data table for ${title}`}
              >
                <thead>
                  <tr role="row">
                    <th 
                      className="border border-border p-2 text-left font-medium"
                      role="columnheader"
                      scope="col"
                    >
                      {xAxisLabel}
                    </th>
                    <th 
                      className="border border-border p-2 text-left font-medium"
                      role="columnheader"
                      scope="col"
                    >
                      {yAxisLabel}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((point, index) => (
                    <tr 
                      key={point.name} 
                      role="row"
                      className={`${
                        index === currentDataIndex 
                          ? 'bg-accent' 
                          : index % 2 === 0 
                            ? 'bg-background' 
                            : 'bg-muted/50'
                      }`}
                    >
                      <td 
                        className="border border-border p-2"
                        role="cell"
                      >
                        {point.name}
                      </td>
                      <td 
                        className="border border-border p-2 font-mono"
                        role="cell"
                      >
                        {typeof point.value === 'number' 
                          ? point.value.toLocaleString() 
                          : point.value
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        
        {/* Hidden announcements for screen readers */}
        <div 
          id={announcementId} 
          className="sr-only" 
          aria-live="polite" 
          aria-atomic="true"
        />
        
        {/* Instructions for screen reader users */}
        <div className="sr-only">
          <p>Chart navigation instructions:</p>
          <ul>
            <li>Press the "Show Table" button to view data in an accessible table format</li>
            <li>In table view, use arrow keys to navigate through data points</li>
            <li>Each data point will be announced when selected</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default AccessibleChart;