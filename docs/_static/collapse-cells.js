// Wrap the input area of every cell tagged "hide-input" in a <details> block
// so the source is collapsed by default; outputs remain visible.
//
// nbconvert (and mkdocs-jupyter) renders cell tags as classes on the cell
// wrapper. Different templates use different prefixes — handle the common
// ones: `celltag_hide-input`, `tag_hide-input`, plain `hide-input`.

(function () {
  function wrapInput(cell) {
    if (cell.dataset.mxFolded === "1") return;
    cell.dataset.mxFolded = "1";

    // Find the input area within this cell (Jupyter / classic / mkdocs-jupyter).
    var input =
      cell.querySelector(".jp-InputArea") ||
      cell.querySelector(".input") ||
      cell.querySelector(".highlight");
    if (!input) return;

    var details = document.createElement("details");
    details.className = "mx-fold";
    var summary = document.createElement("summary");
    details.appendChild(summary);

    input.parentNode.insertBefore(details, input);
    details.appendChild(input);
  }

  function init() {
    var selectors = [
      ".celltag_hide-input",
      ".tag_hide-input",
      ".jp-Cell.hide-input",
      ".cell.hide-input",
    ];
    document.querySelectorAll(selectors.join(",")).forEach(wrapInput);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
