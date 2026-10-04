let conversationId = null;
let ws = null;
let streaming = null;
const toolBlocks = new Map();

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
