// Wrap the full input wrapper of every cell tagged "hide-input" in a
// <details> block so the source is collapsed by default; outputs remain
// visible.
//
// mkdocs-jupyter renders each tagged cell with two nested
// .celltag_hide-input wrappers (outer + inner). The .jp-Cell-inputWrapper
// is only a direct child of the inner one, so the selector below matches
// each input wrapper exactly once — avoiding nested <details>.
(function () {
  function init() {
    var wrappers = document.querySelectorAll(
      ".celltag_hide-input > .jp-Cell-inputWrapper"
    );
    wrappers.forEach(function (wrapper) {
      if (wrapper.dataset.mxFolded === "1") return;
      wrapper.dataset.mxFolded = "1";

      var details = document.createElement("details");
      details.className = "mx-fold";
      var summary = document.createElement("summary");
      details.appendChild(summary);

      wrapper.parentNode.insertBefore(details, wrapper);
      details.appendChild(wrapper);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
