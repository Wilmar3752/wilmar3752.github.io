import { useState, useEffect } from "preact/hooks";
import PhMoonStarsFill from "~icons/ph/moon-stars-fill";
import PhSunFill from "~icons/ph/sun-fill";

export default function ToggleTheme() {
  const [theme, setTheme] = useState(localStorage.getItem("theme") ?? "dark");
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }

    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setIsAnimating(true);
    setTheme(theme === "light" ? "dark" : "light");
    setTimeout(() => setIsAnimating(false), 500);
  };

  return (
    <button
      class="relative w-12 h-12 flex items-center justify-center rounded-xl bg-gradient-to-br from-primary-500/10 to-accent-500/10 hover:from-primary-500/20 hover:to-accent-500/20 border border-primary-500/20 hover:border-primary-500/40 transition-all duration-300 hover:scale-110 hover:shadow-lg group overflow-hidden"
      onClick={toggleTheme}
      aria-label="Toggle theme"
    >
      <div class="absolute inset-0 bg-gradient-to-br from-primary-500/0 to-accent-500/0 group-hover:from-primary-500/10 group-hover:to-accent-500/10 transition-all duration-300"></div>
      
      <div class={`relative transition-transform duration-500 ${isAnimating ? 'rotate-180 scale-0' : 'rotate-0 scale-100'}`}>
        {theme === "dark" ? (
          <PhMoonStarsFill 
            style={{ fontSize: "1.5em" }} 
            class="text-primary-400"
          />
        ) : (
          <PhSunFill 
            style={{ fontSize: "1.5em" }} 
            class="text-amber-500"
          />
        )}
      </div>
      
      <span class="sr-only">Toggle Theme</span>
    </button>
  );
}
