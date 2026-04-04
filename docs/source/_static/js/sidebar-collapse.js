// Collapsible sidebar sections + sidebar toggle for Furo theme
document.addEventListener("DOMContentLoaded", function () {

  // ── Section fold/unfold (caption-level) ──
  var captions = document.querySelectorAll(".sidebar-tree p.caption");

  captions.forEach(function (captionP) {
    var list = captionP.nextElementSibling;
    if (!list || list.tagName !== "UL") return;

    // All sections start expanded (no collapsed class added)

    captionP.addEventListener("click", function () {
      captionP.classList.toggle("collapsed");
    });
  });

  // ── Only expand the current page's parent chain (like XuanCe) ──
  // Do NOT expand all checkboxes; only open the path to the active page.
  var currentPage = document.querySelector(".sidebar-tree .current-page > .reference") ||
                    document.querySelector(".sidebar-tree li.current > .reference");
  if (currentPage) {
    // Walk up from current link and expand each parent <li> checkbox
    var parent = currentPage.closest("li");
    while (parent) {
      var cb = parent.querySelector(":scope > input[type='checkbox']");
      if (cb) cb.checked = true;
      parent = parent.parentElement ? parent.parentElement.closest("li") : null;
    }
    // Ensure the parent caption section is not collapsed
    var ul = currentPage.closest("ul");
    while (ul) {
      var prev = ul.previousElementSibling;
      if (prev && prev.classList.contains("caption")) {
        prev.classList.remove("collapsed");
      }
      ul = ul.parentElement ? ul.parentElement.closest("ul") : null;
    }
    // Scroll into view
    setTimeout(function () {
      currentPage.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 100);
  }

  // ── Desktop sidebar collapse button (inside sidebar brand) ──
  var sidebarBrand = document.querySelector(".sidebar-brand");
  if (!sidebarBrand) return;

  var collapseBtn = document.createElement("button");
  collapseBtn.className = "sidebar-collapse-btn";
  collapseBtn.setAttribute("aria-label", "Collapse sidebar");
  collapseBtn.setAttribute("title", "Collapse sidebar");
  collapseBtn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="3" y="3" width="18" height="18" rx="2"/>' +
    '<line x1="9" y1="3" x2="9" y2="21"/>' +
    '<polyline points="15 9 12 12 15 15"/>' +
    '</svg>';
  sidebarBrand.style.position = "relative";
  sidebarBrand.appendChild(collapseBtn);

  if (localStorage.getItem("mosaic-sidebar-hidden") === "true") {
    document.body.classList.add("sidebar-hidden");
  }

  collapseBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    document.body.classList.add("sidebar-hidden");
    localStorage.setItem("mosaic-sidebar-hidden", "true");
  });

  // Expand button: small square that appears top-left when sidebar is hidden
  var expandBtn = document.createElement("button");
  expandBtn.className = "sidebar-expand-btn";
  expandBtn.setAttribute("aria-label", "Expand sidebar");
  expandBtn.setAttribute("title", "Show sidebar");
  expandBtn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="3" y="3" width="18" height="18" rx="2"/>' +
    '<line x1="9" y1="3" x2="9" y2="21"/>' +
    '<polyline points="13 9 16 12 13 15"/>' +
    '</svg>';
  document.body.appendChild(expandBtn);

  expandBtn.addEventListener("click", function () {
    document.body.classList.remove("sidebar-hidden");
    localStorage.setItem("mosaic-sidebar-hidden", "false");
  });
});
