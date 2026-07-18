const state = { searchFile: null, persons: [], health: null };

const $ = (selector) => document.querySelector(selector);

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden", "error");
  if (error) toast.classList.add("error");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.add("hidden"), 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function loadHealth() {
  state.health = await api("/api/health");
  $("#library-count").textContent = `${state.health.persons} 人 / ${state.health.reference_images} 张参考图`;
  const banner = $("#engine-banner");
  if (!state.health.real_face_recognition) {
    banner.textContent = "当前运行轻量演示引擎，只用于验证界面和数据流程；真实人脸检索请切换至 DeepFace。";
    banner.classList.remove("hidden");
  } else if (!state.health.engine_available) {
    banner.textContent = "已选择真实人脸引擎，但模型文件或运行依赖尚未就绪。请运行 python3 scripts/setup_models.py。";
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

async function loadPersons() {
  state.persons = await api("/api/library/persons?limit=200");
  const list = $("#people-list");
  const select = $("#person-select");
  const quickSelect = $("#quick-person-select");
  const previousQuickPerson = quickSelect.value;
  list.innerHTML = "";
  select.innerHTML = '<option value="">请选择人物</option>';
  quickSelect.innerHTML = '<option value="__new__">＋ 同时创建新人物</option>';

  if (!state.persons.length) {
    list.innerHTML = '<div class="empty-state"><p>人物库为空</p></div>';
  }

  state.persons.forEach((person) => {
    const row = document.createElement("div");
    row.className = "person-row";
    row.innerHTML = `
      <div><strong></strong><small></small></div>
      <span class="status-chip">${person.image_count} 张</span>
      <button class="delete-button" type="button">删除</button>
    `;
    row.querySelector("strong").textContent = person.name;
    row.querySelector("small").textContent = person.external_id || person.id.slice(0, 8);
    row.querySelector("button").addEventListener("click", () => deletePerson(person));
    list.appendChild(row);

    const option = document.createElement("option");
    option.value = person.id;
    option.textContent = `${person.name}（${person.image_count} 张）`;
    select.appendChild(option);

    const quickOption = option.cloneNode(true);
    quickSelect.appendChild(quickOption);
  });
  if ([...quickSelect.options].some((option) => option.value === previousQuickPerson)) {
    quickSelect.value = previousQuickPerson;
  }
  toggleQuickPersonFields();
}

function populateQuickPersons(persons) {
  const select = $("#quick-person-select");
  const previous = select.value;
  select.innerHTML = '<option value="__new__">＋ 同时创建新人物</option>';
  persons.forEach((person) => {
    const option = document.createElement("option");
    option.value = person.id;
    option.textContent = `${person.name}（${person.image_count} 张）`;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
  toggleQuickPersonFields();
}

function toggleQuickPersonFields() {
  const isNew = $("#quick-person-select").value === "__new__";
  $("#quick-new-person").classList.toggle("hidden", !isNew);
  $("#quick-new-person").querySelector('[name="name"]').required = isNew;
}

async function refreshAll() {
  try {
    await Promise.all([loadHealth(), loadPersons()]);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function deletePerson(person) {
  if (!window.confirm(`确定删除“${person.name}”及其所有参考图吗？`)) return;
  try {
    await api(`/api/library/persons/${person.id}`, { method: "DELETE" });
    showToast("人物已删除");
    await refreshAll();
  } catch (error) {
    showToast(error.message, true);
  }
}

function setSearchFile(file) {
  if (!file) return;
  state.searchFile = file;
  const preview = $("#search-preview");
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  $("#upload-placeholder").classList.add("hidden");
  $("#search-button").disabled = false;
}

function renderResults(payload) {
  const container = $("#results");
  container.innerHTML = "";
  $("#search-status").classList.add("hidden");

  if (!payload.faces_detected) {
    container.innerHTML = '<div class="empty-state"><p>没有检测到人脸</p></div>';
    return;
  }

  payload.results.forEach((face) => {
    const section = document.createElement("section");
    section.className = "face-result";
    const heading = document.createElement("h3");
    heading.textContent = `检测到的人脸 ${face.face_number}`;
    section.appendChild(heading);

    face.candidates.forEach((candidate, index) => {
      const row = document.createElement("article");
      row.className = "candidate";
      const percent = Math.max(0, Math.min(100, candidate.aggregate_similarity * 100));
      row.innerHTML = `
        <img src="/api/library/images/${candidate.best_reference_image_id}" alt="" />
        <div><h3></h3><p>排名 #${index + 1} · 匹配 ${candidate.reference_matches} 张参考图</p></div>
        <div class="score">${percent.toFixed(1)}%<small>模型相似度</small></div>
      `;
      row.querySelector("h3").textContent = candidate.name;
      row.querySelector("img").alt = `${candidate.name} 参考图`;
      const sourceUrl = candidate.source_page_url || candidate.source_url;
      if (sourceUrl && /^https?:\/\//i.test(sourceUrl)) {
        const link = document.createElement("a");
        link.className = "source-link";
        link.href = sourceUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = candidate.license_code
          ? `查看图片来源 · ${candidate.license_code}`
          : "查看图片来源";
        row.children[1].appendChild(link);
      }
      section.appendChild(row);
    });
    container.appendChild(section);
  });
}

async function runSearch() {
  if (!state.searchFile) return;
  const button = $("#search-button");
  button.disabled = true;
  button.textContent = "检索中…";
  $("#results").innerHTML = "";
  $("#search-status").classList.remove("hidden");
  $("#search-status p").textContent = "正在提取人脸特征并检索";
  const form = new FormData();
  form.append("image", state.searchFile);
  form.append("top_k", $("#top-k").value);
  try {
    const payload = await api("/api/search", { method: "POST", body: form });
    renderResults(payload);
  } catch (error) {
    $("#search-status p").textContent = error.message;
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "开始检索";
  }
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    $(`#${button.dataset.view}-view`).classList.add("active");
  });
});

const dropZone = $("#drop-zone");
dropZone.addEventListener("click", () => $("#search-file").click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") $("#search-file").click();
});
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  setSearchFile(event.dataTransfer.files[0]);
});
$("#search-file").addEventListener("change", (event) => setSearchFile(event.target.files[0]));
$("#search-button").addEventListener("click", runSearch);

$("#person-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const aliases = String(form.get("aliases") || "").split(",").map((value) => value.trim()).filter(Boolean);
  try {
    await api("/api/library/persons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: form.get("name"), external_id: form.get("external_id") || null, aliases }),
    });
    event.currentTarget.reset();
    showToast("人物已创建");
    await refreshAll();
  } catch (error) {
    showToast(error.message, true);
  }
});

$("#reference-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const personId = form.get("person_id");
  if (!personId) return;
  form.delete("person_id");
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "处理人脸中…";
  try {
    await api(`/api/library/persons/${personId}/images`, { method: "POST", body: form });
    event.currentTarget.reset();
    showToast("参考图已入库");
    await refreshAll();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "上传并生成向量";
  }
});

$("#quick-person-select").addEventListener("change", toggleQuickPersonFields);

$("#quick-person-search").addEventListener("input", (event) => {
  window.clearTimeout(state.quickPersonSearchTimer);
  state.quickPersonSearchTimer = window.setTimeout(async () => {
    const query = event.target.value.trim();
    try {
      const persons = query
        ? await api(`/api/library/persons?limit=50&q=${encodeURIComponent(query)}`)
        : state.persons;
      populateQuickPersons(persons);
    } catch (error) {
      showToast(error.message, true);
    }
  }, 280);
});

$("#quick-source-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const personId = String(form.get("person_id") || "");
  const lines = String(form.get("sources") || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length > 20) {
    showToast("一次最多导入 20 个图片源", true);
    return;
  }

  const licenseCode = String(form.get("license_code") || "").trim() || null;
  let sources;
  try {
    sources = lines.map((line) => {
      const separator = line.indexOf("|");
      const imageUrl = (separator >= 0 ? line.slice(0, separator) : line).trim();
      const sourcePageUrl = (separator >= 0 ? line.slice(separator + 1) : "").trim() || null;
      const parsed = new URL(imageUrl);
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error();
      if (sourcePageUrl) {
        const sourcePage = new URL(sourcePageUrl);
        if (!["http:", "https:"].includes(sourcePage.protocol)) throw new Error();
      }
      return { image_url: imageUrl, source_page_url: sourcePageUrl, license_code: licenseCode };
    });
  } catch (_) {
    showToast("图片源格式有误，请使用 HTTP(S) 图片直链", true);
    return;
  }

  const payload = { sources };
  if (personId === "__new__") {
    payload.person = {
      name: String(form.get("name") || "").trim(),
      external_id: String(form.get("external_id") || "").trim() || null,
      aliases: String(form.get("aliases") || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    };
  } else {
    payload.person_id = personId;
  }

  const button = event.currentTarget.querySelector('button[type="submit"]');
  const result = $("#quick-source-result");
  button.disabled = true;
  button.textContent = "正在下载和生成人脸向量…";
  result.classList.add("hidden");
  try {
    const response = await api("/api/library/quick-source-import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const summary = response.summary;
    result.innerHTML = "";
    result.classList.remove("hidden", "error");
    const title = document.createElement("strong");
    title.textContent = `${response.person.name}：成功 ${summary.imported}，跳过 ${summary.skipped}，失败 ${summary.failed}`;
    result.appendChild(title);
    const detail = document.createElement("span");
    detail.textContent = `本地已关联 ${response.person.image_count} 张参考图，当前索引 ${response.indexed_references} 张。`;
    result.appendChild(detail);
    if (response.failed.length) {
      result.classList.add("error");
      const failures = document.createElement("ul");
      response.failed.forEach((item) => {
        const failure = document.createElement("li");
        failure.textContent = `第 ${item.source_number} 行：${item.error}`;
        failures.appendChild(failure);
      });
      result.appendChild(failures);
    }
    showToast(summary.failed ? "配置完成，部分图片需处理" : "图片源已下载并关联");
    await refreshAll();
  } catch (error) {
    result.textContent = error.message;
    result.classList.remove("hidden");
    result.classList.add("error");
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "一键下载并关联";
  }
});

$("#refresh-library").addEventListener("click", refreshAll);
refreshAll();
