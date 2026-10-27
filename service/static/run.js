// Live progress for a pipeline run: polls the status endpoint and updates the page.
(function () {
  "use strict";
  var section = document.getElementById("progress-section");
  if (!section) { return; }
  var runId = section.getAttribute("data-run");
  var statusEl = document.getElementById("run-status");
  var rowsEl = document.getElementById("stage-rows");
  var logEl = document.getElementById("log-tail");

  function cell(tag, text) {
    var el = document.createElement(tag);
    el.textContent = text;
    return el;
  }

  function render(data) {
    var run = data.run;
    var elapsed = data.elapsed === null ? "" : " (" + Math.round(data.elapsed) + " s)";
    statusEl.textContent = "Status: " + run.status + elapsed;
    statusEl.className = "status status-" + run.status;
    rowsEl.textContent = "";
    if (!data.stages.length) {
      var row = document.createElement("tr");
      var td = cell("td", "Starting the pipeline\u2026");
      td.colSpan = 3;
      row.appendChild(td);
      rowsEl.appendChild(row);
    }
    data.stages.forEach(function (stage) {
      var row = document.createElement("tr");
      row.appendChild(cell("th", stage.stage)).setAttribute("scope", "row");
      row.appendChild(cell("td", stage.status));
      row.appendChild(cell("td", stage.seconds === null ? "" : stage.seconds.toFixed(1)));
      rowsEl.appendChild(row);
    });
    logEl.textContent = data.log_tail.join("\n");
    return run.status === "queued" || run.status === "running";
  }

  function poll() {
    fetch("/api/runs/" + encodeURIComponent(runId), { credentials: "same-origin", cache: "no-store" })
      .then(function (response) { return response.json(); })
      .then(function (data) {
        if (render(data)) {
          window.setTimeout(poll, 2000);
        } else {
          window.setTimeout(function () { window.location.reload(); }, 1000);
        }
      })
      .catch(function () { window.setTimeout(poll, 5000); });
  }

  poll();
})();
