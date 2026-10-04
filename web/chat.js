let conversationId = null;
let ws = null;
let streaming = null;
const toolBlocks = new Map();
let currentDir = "";

function appendMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  if (className === "assistant") {
    div.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } else {
    div.textContent = text;
  }
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView();
}

function toolSection(label, text) {
  const section = document.createElement("div");
  const heading = document.createElement("div");
  heading.className = "tool-label";
  heading.textContent = label;
  const pre = document.createElement("pre");
  pre.textContent = text;
  section.append(heading, pre);
  return section;
}

function addToolCall(data) {
  const details = document.createElement("details");
  details.className = "tool-call pending";
  const summary = document.createElement("summary");
  summary.textContent = data.name;
  details.append(summary, toolSection("Input", JSON.stringify(data.input, null, 2)));
  document.getElementById("messages").appendChild(details);
  details.scrollIntoView();
  toolBlocks.set(data.id, details);
}

function addToolResult(data) {
  const details = toolBlocks.get(data.id);
  if (!details) return;
  toolBlocks.delete(data.id);
  const failed = data.error != null;
  details.classList.replace("pending", failed ? "failed" : "succeeded");
  details.append(toolSection(failed ? "Error" : "Output", failed ? data.error : String(data.output)));
}

function setThinking(visible) {
  document.getElementById("thinking").classList.toggle("hidden", !visible);
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "token") {
      if (!streaming || streaming.index !== data.message) {
        const div = document.createElement("div");
        div.className = "message assistant";
        document.getElementById("messages").appendChild(div);
        streaming = { index: data.message, div, text: "" };
      }
      streaming.text += data.delta;
      streaming.div.innerHTML = DOMPurify.sanitize(marked.parse(streaming.text));
      streaming.div.scrollIntoView();
    } else if (data.type === "tool_call") {
      addToolCall(data);
    } else if (data.type === "tool_result") {
      addToolResult(data);
      setThinking(true);
    } else if (data.type === "final") {
      setThinking(false);
      conversationId = data.conversation_id;
      if (!streaming) appendMessage(data.content, "assistant");
      streaming = null;
      loadFiles();
    } else if (data.type === "error") {
      setThinking(false);
      streaming = null;
      appendMessage(`Error: ${data.message}`, "tool");
    }
  };
}

async function loadMcpStatus() {
  const response = await fetch("/mcp/status");
  const servers = await response.json();
  renderMcpStatus(servers);
}

function renderMcpStatus(servers) {
  const el = document.getElementById("mcp-status");
  el.innerHTML = "";
  for (const server of servers) {
    const div = document.createElement("div");
    const dotSpan = document.createElement("span");
    dotSpan.className = `status-dot ${server.connected ? "connected" : "disconnected"}`;
    div.appendChild(dotSpan);
    div.appendChild(document.createTextNode(server.name));
    if (server.connected && server.tools.length) {
      const list = document.createElement("ul");
      for (const toolName of server.tools) {
        const li = document.createElement("li");
        li.textContent = toolName;
        list.appendChild(li);
      }
      div.appendChild(list);
    }
    el.appendChild(div);
  }
}

function joinPath(dir, name) {
  return dir ? `${dir}/${name}` : name;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function errorDetail(response) {
  try {
    return (await response.json()).detail;
  } catch {
    return response.statusText;
  }
}

async function loadFiles() {
  const response = await fetch(`/files?path=${encodeURIComponent(currentDir || ".")}`);
  if (!response.ok && currentDir) {
    // The folder may have been removed (e.g. by the agent); fall back to the root.
    currentDir = "";
    return loadFiles();
  }
  renderFiles(await response.json());
}

function renderPath() {
  const el = document.getElementById("files-path");
  el.textContent = "";
  const parts = currentDir ? currentDir.split("/") : [];
  const crumbs = [["sandbox", ""], ...parts.map((part, i) => [part, parts.slice(0, i + 1).join("/")])];
  crumbs.forEach(([label, dir], i) => {
    if (i > 0) el.appendChild(document.createTextNode(" / "));
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = label;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      currentDir = dir;
      loadFiles();
    });
    el.appendChild(link);
  });
}

function renderFiles(entries) {
  renderPath();
  const list = document.getElementById("files-list");
  list.textContent = "";
  if (!entries.length) {
    const empty = document.createElement("li");
    empty.className = "files-empty";
    empty.textContent = "(empty folder)";
    list.appendChild(empty);
    return;
  }
  for (const entry of entries) {
    const li = document.createElement("li");
    const path = joinPath(currentDir, entry.name);
    const link = document.createElement("a");
    if (entry.type === "dir") {
      li.className = "dir";
      link.href = "#";
      link.textContent = `${entry.name}/`;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        currentDir = path;
        loadFiles();
      });
    } else {
      li.className = "file";
      link.href = `/files/download?path=${encodeURIComponent(path)}`;
      link.setAttribute("download", entry.name);
      link.textContent = entry.name;
    }
    li.appendChild(link);
    if (entry.type === "file") {
      const size = document.createElement("span");
      size.className = "file-size";
      size.textContent = formatSize(entry.size);
      li.appendChild(size);
    }
    list.appendChild(li);
  }
}

document.getElementById("new-folder").addEventListener("click", async () => {
  const name = window.prompt("Folder name:");
  if (!name) return;
  const response = await fetch("/files/dir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: joinPath(currentDir, name) }),
  });
  if (!response.ok) window.alert(`Could not create folder: ${await errorDetail(response)}`);
  loadFiles();
});

document.getElementById("upload-button").addEventListener("click", () => {
  document.getElementById("upload-input").click();
});

document.getElementById("upload-input").addEventListener("change", async (event) => {
  const input = event.target;
  if (!input.files.length) return;
  const form = new FormData();
  for (const file of input.files) form.append("files", file);
  const response = await fetch(`/files/upload?path=${encodeURIComponent(currentDir || ".")}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) window.alert(`Upload failed: ${await errorDetail(response)}`);
  input.value = "";
  loadFiles();
});

document.getElementById("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const content = input.value.trim();
  if (!content) return;
  appendMessage(content, "user");
  setThinking(true);
  ws.send(JSON.stringify({ type: "message", conversation_id: conversationId, content }));
  input.value = "";
});

connect();
loadMcpStatus();
loadFiles();
