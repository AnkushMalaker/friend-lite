import { useTheme } from "../contexts/ThemeContext";
import { Moon, Sun } from 'lucide-react';

export const ThemeSwitcher = () => {
    const { isDark, toggleTheme } = useTheme();
    return <button
        onClick={toggleTheme}
        className="px-3 py-1 rounded-md bg-gray-200 hover:bg-gray-300 transition-colors"
    >
        {isDark ? <Moon /> : <Sun />}
    </button>;
}