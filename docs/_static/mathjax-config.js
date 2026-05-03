// Enable $...$ as inline math and $$...$$ as display math in MathJax 3.
// Loaded BEFORE the MathJax script tag in mkdocs.yml.
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*\\|noTex",
    processHtmlClass: "arithmatex",
  },
};
