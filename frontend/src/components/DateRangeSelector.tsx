
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { CalendarIcon } from "lucide-react";

interface DateRangeSelectorProps {
  dateRange: { start: string; end: string };
  onDateRangeChange: (dateRange: { start: string; end: string }) => void;
}

const DateRangeSelector = ({ dateRange, onDateRangeChange }: DateRangeSelectorProps) => {
  return (
    <div className="space-y-4">
      <Label className="text-base font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-2">
        <div className="w-2 h-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"></div>
        <CalendarIcon className="w-4 h-4" />
        <span>Analysis Period</span>
      </Label>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="start-date" className="text-xs text-slate-500 dark:text-slate-400">
            Start Date
          </Label>
          <Input
            id="start-date"
            type="date"
            value={dateRange.start}
            onChange={(e) => onDateRangeChange({ ...dateRange, start: e.target.value })}
            className="mt-1 bg-white/80 dark:bg-slate-700/80 border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 transition-colors duration-200 text-slate-900 dark:text-slate-100"
          />
        </div>
        
        <div>
          <Label htmlFor="end-date" className="text-xs text-slate-500 dark:text-slate-400">
            End Date
          </Label>
          <Input
            id="end-date"
            type="date"
            value={dateRange.end}
            onChange={(e) => onDateRangeChange({ ...dateRange, end: e.target.value })}
            className="mt-1 bg-white/80 dark:bg-slate-700/80 border-slate-200 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-400 transition-colors duration-200 text-slate-900 dark:text-slate-100"
          />
        </div>
      </div>
    </div>
  );
};

export default DateRangeSelector;
