const boardEl = document.getElementById("board");
const filesEl = document.getElementById("files");
const sublineEl = document.getElementById("subline");
const turnEl = document.getElementById("turn");
const stateEl = document.getElementById("state");
const lastMoveEl = document.getElementById("last-move");
const legalCountEl = document.getElementById("legal-count");
const historyEl = document.getElementById("history");
const fenEl = document.getElementById("fen");
const errorEl = document.getElementById("error");
const newGameBtn = document.getElementById("new-game");
const computerBtn = document.getElementById("computer");
const refreshBtn = document.getElementById("refresh");
const flipBoardBtn = document.getElementById("flip-board");

const pieces = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

let snapshot = null;
let selected = null;
let legalTargets = new Set();
let flipped = false;
let requestInFlight = false;

function boardSquares() {
  const squares = [];
  const ranks = flipped ? [1, 2, 3, 4, 5, 6, 7, 8] : [8, 7, 6, 5, 4, 3, 2, 1];
  const files = flipped ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  for (const rank of ranks) {
    for (const file of files) {
      squares.push("abcdefgh"[file] + rank);
    }
  }
  return squares;
}

function fileLabels() {
  return (flipped ? "hgfedcba" : "abcdefgh").split("");
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
  filesEl.innerHTML = "";
  for (const file of fileLabels()) {
    const label = document.createElement("span");
    label.textContent = file;
    filesEl.appendChild(label);
  }

  for (const square of boardSquares()) {
    const file = square.charCodeAt(0) - 97;
    const rank = Number(square[1]) - 1;
    const button = document.createElement("button");
    button.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;
    if (selected === square) button.classList.add("selected");
    if (legalTargets.has(square)) button.classList.add("legal");
    if (snapshot?.last_move?.slice(0, 2) === square || snapshot?.last_move?.slice(2, 4) === square) {
      button.classList.add("last");
    }
    button.dataset.square = square;
    button.disabled = Boolean(snapshot?.is_busy || requestInFlight);
    const piece = pieces[snapshot?.board?.[square]] || "";
    if (piece) {
      const pieceEl = document.createElement("span");
      pieceEl.className = `piece ${snapshot.board[square] === snapshot.board[square].toUpperCase() ? "white-piece" : "black-piece"}`;
      pieceEl.textContent = piece;
      button.appendChild(pieceEl);
    }
    const coord = document.createElement("span");
    coord.className = "coord";
    coord.textContent = square;
    button.appendChild(coord);
    button.addEventListener("click", () => onSquare(square));
    boardEl.appendChild(button);
  }
}

async function onSquare(square) {
  if (!snapshot || snapshot.is_busy || requestInFlight) return;
  const piece = snapshot.board[square];
  if (!selected) {
    if (!piece) return;
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
  if (piece && !legalTargets.has(square)) {
    selected = square;
    legalTargets = legalDests(square);
    renderBoard();
    return;
  }
  if (!legalTargets.has(square)) {
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

function stateText() {
  const status = snapshot.status || {};
  if (snapshot.is_busy || requestInFlight) return "Busy";
  if (status.is_checkmate) return "Checkmate";
  if (status.is_stalemate) return "Stalemate";
  if (status.is_game_over) return "Game over";
  if (status.is_check) return "Check";
  return "Ready";
}

function renderHistory() {
  historyEl.innerHTML = "";
  const moves = snapshot.move_history_san || [];
  if (!moves.length) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = "No moves yet";
    historyEl.appendChild(empty);
    return;
  }
  moves.forEach((move, index) => {
    const chip = document.createElement("span");
    chip.className = "move-chip";
    chip.textContent = `${index + 1}. ${move}`;
    historyEl.appendChild(chip);
  });
}

function renderSnapshot() {
  turnEl.textContent = snapshot.turn === "white" ? "White" : "Black";
  stateEl.textContent = stateText();
  stateEl.dataset.state = stateEl.textContent.toLowerCase().replaceAll(" ", "-");
  sublineEl.textContent = snapshot.error || `${stateEl.textContent} · ${snapshot.turn === "white" ? "White" : "Black"} to move`;
  lastMoveEl.textContent = snapshot.last_move || "-";
  legalCountEl.textContent = String((snapshot.legal_moves || []).length);
  fenEl.textContent = snapshot.fen || "-";
  renderHistory();
  errorEl.textContent = snapshot.error || "";
  newGameBtn.disabled = Boolean(snapshot.is_busy || requestInFlight);
  computerBtn.disabled = Boolean(snapshot.is_busy || requestInFlight);
  refreshBtn.disabled = Boolean(requestInFlight);
  renderBoard();
}

async function loadSnapshot() {
  const response = await fetch("/api/snapshot");
  snapshot = await response.json();
  renderSnapshot();
}

async function postJson(url, payload = {}) {
  requestInFlight = true;
  if (snapshot) renderSnapshot();
  try {
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
  } finally {
    requestInFlight = false;
    if (snapshot) renderSnapshot();
  }
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

refreshBtn.addEventListener("click", async () => {
  await loadSnapshot();
});

flipBoardBtn.addEventListener("click", () => {
  flipped = !flipped;
  renderBoard();
});

loadSnapshot();
setInterval(() => {
  if (snapshot?.is_busy) loadSnapshot();
}, 500);
