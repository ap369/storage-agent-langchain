let token = null;
let conversationId = null;
let ws = null;

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

function setThinking(visible) {
  document.getElementById("thinking").classList.toggle("hidden", !visible);
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
  ws.onopen = () => ws.send(JSON.stringify({ token }));
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "tool_call") {
      appendMessage(`→ ${data.name}(${JSON.stringify(data.input)})`, "tool");
    } else if (data.type === "tool_result") {
      appendMessage(`← ${data.name}: ${data.output}`, "tool");
      setThinking(true);
    } else if (data.type === "final") {
      setThinking(false);
      conversationId = data.conversation_id;
      appendMessage(data.content, "assistant");
    } else if (data.type === "error") {
      setThinking(false);
      appendMessage(`Error: ${data.message}`, "tool");
    }
  };
}

async function loadMcpStatus() {
  const response = await fetch("/mcp/status", { headers: { Authorization: `Bearer ${token}` } });
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

token = window.prompt("API token:");
connect();
loadMcpStatus();
