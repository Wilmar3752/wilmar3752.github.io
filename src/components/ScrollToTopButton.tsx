export default function ScrollToTopButton() {
  function handleOnClick() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
  }

  return (
    <button
      onClick={handleOnClick}
      className="group relative overflow-hidden rounded-lg px-4 py-2 bg-gradient-to-r from-primary-600 to-accent-500 text-white font-semibold text-xs uppercase shadow-md hover:shadow-xl hover:scale-105 transition-all duration-300"
    >
      <span className="relative z-10 flex items-center gap-2">
        <svg className="w-4 h-4 transform group-hover:-translate-y-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
        Back to top
      </span>
    </button>
  );
}
