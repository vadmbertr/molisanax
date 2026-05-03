// Wrap the input wrapper of every cell tagged "hide-input" in a plain
// <div> that can be toggled by a button. We deliberately do NOT use
// <details>/<summary> because Material for MkDocs applies admonition-like
// styling (background, border, padding) to those elements.
(function () {
  function makeToggle(wrapper) {
    if (wrapper.dataset.mxFolded === "1") return;
    wrapper.dataset.mxFolded = "1";

    var fold = document.createElement("div");
    fold.className = "mx-fold mx-fold--closed";

    var button = document.createElement("button");
    button.type = "button";
    button.className = "mx-fold-toggle";
    button.textContent = "▸ Show code";
    button.setAttribute("aria-expanded", "false");

    var body = document.createElement("div");
    body.className = "mx-fold-body";

    button.addEventListener("click", function () {
      var open = fold.classList.toggle("mx-fold--open");
      fold.classList.toggle("mx-fold--closed", !open);
      button.textContent = (open ? "▾ Hide code" : "▸ Show code");
      button.setAttribute("aria-expanded", open ? "true" : "false");
    });

    wrapper.parentNode.insertBefore(fold, wrapper);
    fold.appendChild(button);
    fold.appendChild(body);
    body.appendChild(wrapper);
  }

  function init() {
    document
      .querySelectorAll(".celltag_hide-input > .jp-Cell-inputWrapper")
      .forEach(makeToggle);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
