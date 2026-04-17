(function () {
  const $ = (id) => document.getElementById(id);

  const apiAnalyze =
    typeof window.__GAP_API_ANALYZE__ === "string" && window.__GAP_API_ANALYZE__
      ? window.__GAP_API_ANALYZE__
      : "/api/analyze";

  function csrfToken() {
    const m = document.querySelector('meta[name="csrf-token"]');
    return m ? m.getAttribute("content") || "" : "";
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function renderParsed(p) {
    const el = $("content");
    el.innerHTML = "";

    if (p == null || typeof p !== "object" || Array.isArray(p)) {
      el.innerHTML =
        "<p class=\"fine-print\">The reply wasn’t in the shape we expected. If you see raw text below in the server response, try again.</p>";
      return;
    }

    const fixes = (p.priority_fixes || []).slice().sort((a, b) => (a.rank || 0) - (b.rank || 0));
    const skills = p.missing_skills || [];
    const kws = p.missing_keywords || [];

    if (fixes.length) {
      const sheet = document.createElement("div");
      sheet.className = "sheet";
      sheet.innerHTML = "<h3>fix these first (sorry)</h3><ol class=\"fix-list\"></ol>";
      const ol = sheet.querySelector("ol");
      fixes.forEach((f) => {
        const li = document.createElement("li");
        li.innerHTML =
          "<span class=\"action\">" + esc(f.action || f.do_this || "") + "</span>" +
          "<div class=\"why\">" + esc(f.why || f.because || "") + "</div>";
        ol.appendChild(li);
      });
      el.appendChild(sheet);
    }

    if (skills.length) {
      const sheet = document.createElement("div");
      sheet.className = "sheet";
      sheet.innerHTML = "<h3>skills they actually name · you don’t</h3><ul class=\"tag-list\"></ul>";
      const ul = sheet.querySelector("ul");
      skills.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = esc(s.skill || "") + "<em>" + esc(s.note || s.evidence_from_posting || "") + "</em>";
        ul.appendChild(li);
      });
      el.appendChild(sheet);
    }

    if (kws.length) {
      const sheet = document.createElement("div");
      sheet.className = "sheet";
      sheet.innerHTML = "<h3>words they’ll ctrl+f · you’re thin on</h3><ul class=\"tag-list\"></ul>";
      const ul = sheet.querySelector("ul");
      kws.forEach((k) => {
        const li = document.createElement("li");
        li.innerHTML = "“" + esc(k.phrase || "") + "”<em>" + esc(k.gap || k.where_in_posting || "") + "</em>";
        ul.appendChild(li);
      });
      el.appendChild(sheet);
    }

    if (!fixes.length && !skills.length && !kws.length) {
      el.innerHTML =
        "<p class=\"fine-print\">Nothing to tally — paste more of the notice or your page, or try again.</p>";
    }
  }

  function setPdfChip(file) {
    const chip = $("file-chip");
    const name = $("file-name");
    if (!file || !file.name) {
      chip.classList.add("hidden");
      name.textContent = "";
      return;
    }
    name.textContent = file.name;
    chip.classList.remove("hidden");
  }

  const pdfInput = $("resume_pdf");
  const dropzone = $("dropzone");

  $("clear-pdf").addEventListener("click", (e) => {
    e.preventDefault();
    pdfInput.value = "";
    setPdfChip(null);
  });

  pdfInput.addEventListener("change", () => {
    const f = pdfInput.files && pdfInput.files[0];
    setPdfChip(f || null);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-drag");
    });
  });

  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-drag");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    if (f.type !== "application/pdf") {
      $("error").textContent = "Only PDF files are accepted for the résumé slot.";
      $("error").classList.remove("hidden");
      return;
    }
    try {
      const dt = new DataTransfer();
      dt.items.add(f);
      pdfInput.files = dt.files;
      setPdfChip(f);
    } catch {
      $("error").textContent = "Couldn’t attach that file.";
      $("error").classList.remove("hidden");
    }
  });

  $("run").addEventListener("click", async () => {
    const job_posting = $("job").value;
    const resume = $("resume").value;
    const btn = $("run");
    const results = $("results");
    const err = $("error");
    const content = $("content");

    err.classList.add("hidden");
    err.textContent = "";
    content.innerHTML = "";
    results.classList.remove("hidden");

    btn.disabled = true;
    btn.textContent = "one sec…";

    const tok = csrfToken();
    const headers = { "X-CSRF-Token": tok };

    const pdfFile = pdfInput.files && pdfInput.files[0];
    let body;
    let fetchHeaders = { ...headers };

    if (pdfFile) {
      const fd = new FormData();
      fd.append("job_posting", job_posting);
      fd.append("resume", resume);
      fd.append("resume_pdf", pdfFile, pdfFile.name || "resume.pdf");
      body = fd;
    } else {
      fetchHeaders["Content-Type"] = "application/json";
      body = JSON.stringify({ job_posting, resume });
    }

    try {
      const r = await fetch(apiAnalyze, {
        method: "POST",
        headers: fetchHeaders,
        body,
        credentials: "same-origin",
      });

      const rawText = await r.text();
      let data;
      try {
        data = JSON.parse(rawText);
      } catch {
        err.textContent = "Couldn’t read the server reply (not JSON). First lines: " + rawText.slice(0, 280);
        err.classList.remove("hidden");
        return;
      }

      if (!r.ok) {
        let msg = data.error || "Something went wrong.";
        if (data.detail) {
          msg += " " + (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
        }
        err.textContent = msg;
        err.classList.remove("hidden");
        return;
      }

      if (data.parsed) {
        renderParsed(data.parsed);
      } else if (data.raw_text) {
        content.innerHTML =
          "<p class=\"fine-print\">The model wandered off the form. Raw ink:</p><pre class=\"raw\">" +
          esc(data.raw_text) +
          "</pre>";
      } else {
        content.innerHTML = "<p class=\"fine-print\">Empty page back from the wire.</p>";
      }
    } catch (e) {
      err.textContent = "Network: " + (e && e.message ? e.message : String(e));
      err.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      btn.textContent = "do the thing";
    }
  });
})();
