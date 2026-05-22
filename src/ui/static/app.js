const boardEl = document.getElementById("board");
const turnEl = document.getElementById("turn");
const stateEl = document.getElementById("state");
const historyEl = document.getElementById("history");
const errorEl = document.getElementById("error");
const newGameBtn = document.getElementById("new-game");
const computerBtn = document.getElementById("computer");

const pieces = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

let snapshot = null;
let selected = null;
let legalTargets = new Set();

function squaresForWhiteView() {
  const squares = [];
  for (let rank = 1; rank <= 8; rank += 1) {
    for (let file = 7; file >= 0; file -= 1) {
      squares.push("abcdefgh"[file] + rank);
    }
  }
  return squares.reverse();
}

function legalDests(square) {
  const dests = new Set();
  for (const move of snapshot.legal_moves || []) {
    if (move.slice(0, 2) === square) dests.add(move.slice(2, 4));
  }
  return dests;
}

function renderBoard() {
  boardEl.innerHTML = "";
  for (const square of squaresForWhiteView()) {
    const file = square.charCodeAt(0) - 97;
    const rank = Number(square[1]) - 1;
    const button = document.createElement("button");
    button.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;
    if (selected === square) button.classList.add("selected");
    if (legalTargets.has(square)) button.classList.add("legal");
    button.dataset.square = square;
    button.disabled = Boolean(snapshot?.is_busy);
    button.textContent = pieces[snapshot?.board?.[square]] || "";
    const coord = document.createElement("span");
    coord.className = "coord";
    coord.textContent = square;
    button.appendChild(coord);
    button.addEventListener("click", () => onSquare(square));
    boardEl.appendChild(button);
  }
}

async function onSquare(square) {
  if (!snapshot || snapshot.is_busy) return;
  if (!selected) {
    if (!snapshot.board[square]) return;
    selected = square;
    legalTargets = legalDests(square);
    renderBoard();
    return;
  }
  if (selected === square) {
    selected = null;
    legalTargets = new Set();
    renderBoard();
    return;
  }
  await postJson("/api/move", { src: selected, dst: square });
  selected = null;
  legalTargets = new Set();
  await loadSnapshot();
}

function renderSnapshot() {
  turnEl.textContent = snapshot.turn === "white" ? "White" : "Black";
  const status = snapshot.status || {};
  stateEl.textContent = status.is_checkmate ? "Checkmate" : status.is_check ? "Check" : snapshot.is_busy ? "Busy" : "Ready";
  historyEl.textContent = (snapshot.move_history_san || []).join(" ");
  errorEl.textContent = snapshot.error || "";
  newGameBtn.disabled = Boolean(snapshot.is_busy);
  computerBtn.disabled = Boolean(snapshot.is_busy);
  renderBoard();
}

async function loadSnapshot() {
  const response = await fetch("/api/snapshot");
  snapshot = await response.json();
  renderSnapshot();
}

async function postJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (data.snapshot) snapshot = data.snapshot;
  else snapshot = data;
  renderSnapshot();
  return data;
}

newGameBtn.addEventListener("click", async () => {
  selected = null;
  legalTargets = new Set();
  await postJson("/api/new-game");
});

computerBtn.addEventListener("click", async () => {
  selected = null;
  legalTargets = new Set();
  await postJson("/api/let-computer-play");
});

loadSnapshot();
setInterval(() => {
  if (snapshot?.is_busy) loadSnapshot();
}, 500);
