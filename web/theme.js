// Light / dark theme for the web UI.
// First visit follows the OS setting; the toggle then remembers your choice.
// Gradio keeps a `dark` class on <body>, so that is the one switch we drive.
(function () {
  var KEY = "audiobook-theme";
  var params = new URLSearchParams(location.search);
  var media = matchMedia("(prefers-color-scheme: dark)");

  function saved() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  function wanted() {
    var fromUrl = params.get("__theme");  // Gradio's own ?__theme=dark works too
    if (fromUrl === "dark" || fromUrl === "light") return fromUrl;
    return saved() || (media.matches ? "dark" : "light");
  }

  function apply() {
    var mode = wanted();
    document.documentElement.dataset.theme = mode;  // paints the page background before Gradio loads
    if (document.body) document.body.classList.toggle("dark", mode === "dark");
  }

  apply();
  document.addEventListener("DOMContentLoaded", function () {
    apply();
    // Gradio also sets the class (from the OS setting); put ours back on top
    new MutationObserver(apply).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  });
  media.addEventListener("change", apply);

  document.addEventListener("click", function (event) {
    if (!event.target.closest("#theme-toggle")) return;
    var next = document.body.classList.contains("dark") ? "light" : "dark";
    try { localStorage.setItem(KEY, next); } catch (e) {}
    params.delete("__theme");
    apply();
  });
})();
