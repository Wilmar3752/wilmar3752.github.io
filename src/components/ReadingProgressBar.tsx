import { useEffect, useState } from "preact/hooks";

export default function ReadingProgressBar() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const updateProgress = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollProgress = (scrollTop / docHeight) * 100;
      setProgress(scrollProgress);
    };

    window.addEventListener("scroll", updateProgress);
    updateProgress(); // Initial calculation

    return () => window.removeEventListener("scroll", updateProgress);
  }, []);

  return (
    <div class="fixed top-0 left-0 right-0 h-1 bg-slate-200 dark:bg-slate-800 z-50">
      <div
        class="h-full bg-gradient-to-r from-primary-600 via-accent-500 to-secondary-500 transition-all duration-150 ease-out shadow-lg shadow-primary-500/50"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
