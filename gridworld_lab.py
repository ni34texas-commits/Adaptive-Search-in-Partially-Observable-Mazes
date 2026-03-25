"""
Part 0 - Gridworld Lab
======================
Generates 30 maze environments (51x51) using DFS with random tie-breaking.
- 30% blocked, 70% unblocked probability
- Saves all grids to JSON + PNG images
- Interactive matplotlib viewer

Requirements:
    pip install matplotlib numpy

Run:
    python gridworld_lab.py
"""

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.colors import ListedColormap

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE   = 51
NUM_WORLDS  = 30
BLOCK_PROB  = 0.30 # 30% chance a cell is blocked
DIRS4       = [(-1,0),(1,0),(0,-1),(0,1)]   # N S W E
OUTPUT_DIR  = "gridworlds"
JSON_PATH   = os.path.join(OUTPUT_DIR, "gridworlds_30.json")
IMG_DIR     = os.path.join(OUTPUT_DIR, "images")

# ─── Maze Generation (DFS, exact spec) ────────────────────────────────────────
def generate_maze(seed: int) -> np.ndarray:
    """
    DFS maze generation as specified:
    1. All cells start unvisited.
    2. Start from random cell → mark unblocked, push to stack.
    3. Pick random unvisited neighbor:
       - 30% → blocked (do NOT push)
       - 70% → unblocked (push to stack)
    4. Dead end → backtrack (pop stack).
    5. Stack empty but unvisited remain → jump to any unvisited cell.
    6. Repeat until all cells visited.
    """
    rng = random.Random(seed)
    rows = cols = GRID_SIZE

    grid    = np.ones((rows, cols), dtype=np.int8)   # all blocked initially
    visited = np.zeros((rows, cols), dtype=bool) # all visted

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    #This function finds all unvisited neighbors and shuffles them randomly — this is the random tie-breaking
    def unvisited_neighbors(r, c):
        dirs = DIRS4[:]
        rng.shuffle(dirs) #random tie breaking
        return [(r+dr, c+dc) for dr,dc in dirs
                if in_bounds(r+dr, c+dc) and not visited[r+dr][c+dc]]

    # Flat list of all cells for restart scanning
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)
    unvisited_set = set(all_cells)

    # random Start cell — always unblocked and push it to stack
    sr, sc = rng.randint(0, rows-1), rng.randint(0, cols-1) 
    visited[sr][sc] = True
    grid[sr][sc]    = 0
    unvisited_set.discard((sr, sc))
    stack = [(sr, sc)]

    while unvisited_set: #keeps going until set is empty
        if not stack:
            # Restart from any unvisited cell
            nr, nc = next(iter(unvisited_set)) # jump to any unvisited cell
            visited[nr][nc] = True
            unvisited_set.discard((nr, nc))
            if rng.random() >= BLOCK_PROB: #Block_prob =.30
                grid[nr][nc] = 0 #70% unblocked
                stack.append((nr, nc)) # only push if unblocked
            else:
                grid[nr][nc] = 1 #30% block probabily
            continue

        r, c = stack[-1]
        neighbors = unvisited_neighbors(r, c)

        if not neighbors:
            stack.pop()   # dead end → backtrack
        else:
            nr, nc = neighbors[0]   # already shuffled
            visited[nr][nc] = True
            unvisited_set.discard((nr, nc))
            if rng.random() < BLOCK_PROB:
                grid[nr][nc] = 1    # blocked, don't push
            else:
                grid[nr][nc] = 0    # unblocked, push
                stack.append((nr, nc))

    return grid


def place_agent_target(grid: np.ndarray, rng: random.Random):
    """Place agent and target on random distinct unblocked cells."""
    free = list(zip(*np.where(grid == 0)))
    if len(free) < 2:
        return (0, 0), (GRID_SIZE-1, GRID_SIZE-1)
    agent  = rng.choice(free)
    target = rng.choice([c for c in free if c != agent])
    return tuple(map(int, agent)), tuple(map(int, target))


def build_world(world_id: int, seed: int) -> dict:
    """Build a complete world dictionary."""
    grid = generate_maze(seed)
    rng  = random.Random(seed ^ 0xDEADBEEF)
    agent, target = place_agent_target(grid, rng)
    blocked   = int(np.sum(grid == 1))
    unblocked = int(np.sum(grid == 0))
    total     = GRID_SIZE * GRID_SIZE
    manhattan = abs(agent[0]-target[0]) + abs(agent[1]-target[1])
    return {
        "id":        world_id,
        "seed":      seed,
        "grid":      grid,
        "agent":     agent,
        "target":    target,
        "blocked":   blocked,
        "unblocked": unblocked,
        "block_rate": round(blocked / total * 100, 1),
        "manhattan": manhattan,
    }


def generate_all_worlds(seeds=None) -> list:
    """Generate all 30 worlds."""
    worlds = []
    for i in range(NUM_WORLDS):
        seed = seeds[i] if seeds else (i + 1) * 0x9E3779B9 & 0xFFFFFFFF
        worlds.append(build_world(i + 1, seed))
        print(f"  Generated world {i+1:2d}/30  (seed={seed})", end="\r")
    print(f"\n  Done! All {NUM_WORLDS} worlds generated.")
    return worlds


# ─── Save / Load ──────────────────────────────────────────────────────────────
def save_worlds(worlds: list):
    """Save all worlds to JSON and PNG files."""
    os.makedirs(IMG_DIR, exist_ok=True)

    # JSON
    serializable = []
    for w in worlds:
        serializable.append({
            "id":         w["id"],
            "seed":       w["seed"],
            "agent":      list(w["agent"]),
            "target":     list(w["target"]),
            "blocked":    w["blocked"],
            "unblocked":  w["unblocked"],
            "block_rate": w["block_rate"],
            "manhattan":  w["manhattan"],
            "grid":       w["grid"].tolist(),
        })
    with open(JSON_PATH, "w") as f:
        json.dump(serializable, f)
    print(f"  Saved JSON → {JSON_PATH}")

    # PNGs
    for w in worlds:
        save_world_image(w)
    print(f"  Saved {NUM_WORLDS} PNG images → {IMG_DIR}/")


def save_world_image(w: dict):
    """Save a single world as a PNG image."""
    fig, ax = _make_grid_figure(w, figsize=(5, 5))
    path = os.path.join(IMG_DIR, f"gridworld_{w['id']:02d}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight", facecolor="#0d0d18")
    plt.close(fig)


def load_worlds(path=JSON_PATH) -> list:
    """Load worlds from JSON file."""
    with open(path) as f:
        data = json.load(f)
    worlds = []
    for d in data:
        d["grid"]   = np.array(d["grid"], dtype=np.int8)
        d["agent"]  = tuple(d["agent"])
        d["target"] = tuple(d["target"])
        worlds.append(d)
    print(f"  Loaded {len(worlds)} worlds from {path}")
    return worlds


# ─── Drawing ──────────────────────────────────────────────────────────────────
CMAP = ListedColormap(["#dde0f0", "#1c1c30"])   # 0=unblocked(light), 1=blocked(dark)

def _make_grid_figure(w: dict, figsize=(6, 6)):
    """Create a matplotlib figure for a world."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d18")
    ax.set_facecolor("#0d0d18")

    ax.imshow(w["grid"], cmap=CMAP, vmin=0, vmax=1,
              origin="upper", interpolation="nearest")

    # Agent
    ar, ac = w["agent"]
    ax.plot(ac, ar, "o", color="#00e5b0", markersize=9, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)
    ax.text(ac, ar, "A", color="white", fontsize=5, ha="center", va="center",
            fontweight="bold", zorder=6)

    # Target
    tr, tc = w["target"]
    ax.plot(tc, tr, "o", color="#ff5370", markersize=9, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)
    ax.text(tc, tr, "T", color="white", fontsize=5, ha="center", va="center",
            fontweight="bold", zorder=6)

    ax.set_title(f"World #{w['id']}  |  Block rate: {w['block_rate']}%  |  Manhattan: {w['manhattan']}",
                 color="#ccd0e8", fontsize=9, pad=6, fontfamily="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333350")

    return fig, ax


# ─── Interactive Viewer ───────────────────────────────────────────────────────
class GridWorldViewer:
    def __init__(self, worlds: list):
        self.worlds   = worlds
        self.idx      = 0
        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 7), facecolor="#0d0d18")
        self.fig.canvas.manager.set_window_title("Gridworld Laboratory — Part 0")

        # ── Left: Grid display ───────────────────────────────────────────────
        self.ax_grid = self.fig.add_axes([0.02, 0.12, 0.52, 0.82])
        self.ax_grid.set_facecolor("#0d0d18")

        # ── Right: Stats panel ───────────────────────────────────────────────
        self.ax_stats = self.fig.add_axes([0.57, 0.35, 0.40, 0.58])
        self.ax_stats.set_facecolor("#0d0d18")
        self.ax_stats.axis("off")

        # ── Bar chart: all block rates ───────────────────────────────────────
        self.ax_bar = self.fig.add_axes([0.57, 0.12, 0.40, 0.20])
        self.ax_bar.set_facecolor("#0d0d18")

        # ── Buttons ──────────────────────────────────────────────────────────
        ax_prev   = self.fig.add_axes([0.02, 0.03, 0.08, 0.05])
        ax_next   = self.fig.add_axes([0.11, 0.03, 0.08, 0.05])
        ax_regen  = self.fig.add_axes([0.22, 0.03, 0.14, 0.05])
        ax_reall  = self.fig.add_axes([0.37, 0.03, 0.16, 0.05])
        ax_save   = self.fig.add_axes([0.57, 0.03, 0.18, 0.05])
        ax_savall = self.fig.add_axes([0.77, 0.03, 0.20, 0.05])

        def btn_style(ax, label, color):
            b = Button(ax, label, color=color, hovercolor="#2a2a40")
            b.label.set_fontfamily("monospace")
            b.label.set_fontsize(8)
            b.label.set_color("#e0e0f0")
            return b

        self.btn_prev   = btn_style(ax_prev,   "◀ PREV",         "#1a1a2e")
        self.btn_next   = btn_style(ax_next,   "NEXT ▶",         "#1a1a2e")
        self.btn_regen  = btn_style(ax_regen,  "↺ REGEN THIS",   "#1a2a1a")
        self.btn_reall  = btn_style(ax_reall,  "↺ REGEN ALL 30", "#2a1a1a")
        self.btn_save   = btn_style(ax_save,   "⬇ SAVE THIS PNG","#1a1a2a")
        self.btn_savall = btn_style(ax_savall, "⬇ SAVE ALL + JSON","#1a1a2a")

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_regen.on_clicked(self._regen_current)
        self.btn_reall.on_clicked(self._regen_all)
        self.btn_save.on_clicked(self._save_current)
        self.btn_savall.on_clicked(self._save_all)

        self._draw()
        plt.show()

    def _draw(self):
        w = self.worlds[self.idx]

        # ── Grid ─────────────────────────────────────────────────────────────
        self.ax_grid.cla()
        self.ax_grid.set_facecolor("#0d0d18")
        self.ax_grid.imshow(w["grid"], cmap=CMAP, vmin=0, vmax=1,
                            origin="upper", interpolation="nearest")
        ar, ac = w["agent"]
        tr, tc = w["target"]
        self.ax_grid.plot(ac, ar, "o", color="#00e5b0", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        self.ax_grid.text(ac, ar, "A", color="white", fontsize=6, ha="center",
                          va="center", fontweight="bold", zorder=6)
        self.ax_grid.plot(tc, tr, "o", color="#ff5370", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        self.ax_grid.text(tc, tr, "T", color="white", fontsize=6, ha="center",
                          va="center", fontweight="bold", zorder=6)

        self.ax_grid.set_title(
            f"WORLD #{w['id']} / {NUM_WORLDS}   |   Seed: {w['seed']}",
            color="#00e5b0", fontsize=10, fontfamily="monospace", pad=8)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])

        # Legend
        patches = [
            mpatches.Patch(color="#1c1c30", label="Blocked"),
            mpatches.Patch(color="#dde0f0", label="Unblocked"),
            mpatches.Patch(color="#00e5b0", label="Agent (A)"),
            mpatches.Patch(color="#ff5370", label="Target (T)"),
        ]
        self.ax_grid.legend(handles=patches, loc="lower left", fontsize=7,
                            facecolor="#1a1a2e", edgecolor="#333350",
                            labelcolor="white", framealpha=0.9)

        # ── Stats ─────────────────────────────────────────────────────────────
        self.ax_stats.cla()
        self.ax_stats.set_facecolor("#0d0d18")
        self.ax_stats.axis("off")

        title_props = dict(color="#7c6af7", fontfamily="monospace",
                           fontsize=9, fontweight="bold")
        val_props   = dict(color="#ccd0e8", fontfamily="monospace", fontsize=9)
        key_props   = dict(color="#6668a0", fontfamily="monospace", fontsize=9)

        self.ax_stats.text(0.0, 1.02, "STATISTICS", transform=self.ax_stats.transAxes,
                           **title_props)

        rows = [
            ("Grid Size",    f"{GRID_SIZE} × {GRID_SIZE}"),
            ("Total Cells",  f"{GRID_SIZE*GRID_SIZE:,}"),
            ("Unblocked",    f"{w['unblocked']:,}"),
            ("Blocked",      f"{w['blocked']:,}"),
            ("Block Rate",   f"{w['block_rate']}%"),
            ("Agent Pos",    f"({w['agent'][0]}, {w['agent'][1]})"),
            ("Target Pos",   f"({w['target'][0]}, {w['target'][1]})"),
            ("Manhattan",    str(w["manhattan"])),
            ("Seed",         str(w["seed"])),
        ]
        for i, (k, v) in enumerate(rows):
            y = 0.92 - i * 0.10
            self.ax_stats.text(0.0, y, k, transform=self.ax_stats.transAxes, **key_props)
            self.ax_stats.text(1.0, y, v, transform=self.ax_stats.transAxes,
                               ha="right", **val_props)
            self.ax_stats.plot([0, 1], [y - 0.02, y - 0.02], color="#222235",
                               linewidth=0.5, transform=self.ax_stats.transAxes)

        # ── Bar chart ─────────────────────────────────────────────────────────
        self.ax_bar.cla()
        self.ax_bar.set_facecolor("#0d0d18")
        rates = [w2["block_rate"] for w2 in self.worlds]
        colors = ["#00e5b0" if i == self.idx else "#7c6af740"
                  for i in range(NUM_WORLDS)]
        self.ax_bar.bar(range(NUM_WORLDS), rates, color=colors, width=0.8)
        self.ax_bar.set_xlim(-0.5, NUM_WORLDS - 0.5)
        self.ax_bar.set_ylim(0, 55)
        self.ax_bar.axhline(30, color="#ff537040", linewidth=1, linestyle="--")
        self.ax_bar.set_xticks([])
        self.ax_bar.set_ylabel("Block %", color="#6668a0",
                               fontfamily="monospace", fontsize=7)
        self.ax_bar.tick_params(colors="#6668a0", labelsize=7)
        self.ax_bar.set_title("All 30 Block Rates", color="#6668a0",
                              fontfamily="monospace", fontsize=8, pad=3)
        for spine in self.ax_bar.spines.values():
            spine.set_edgecolor("#222235")

        self.fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _prev(self, _):
        self.idx = max(0, self.idx - 1)
        self._draw()

    def _next(self, _):
        self.idx = min(NUM_WORLDS - 1, self.idx + 1)
        self._draw()

    def _regen_current(self, _):
        new_seed = random.randint(0, 0xFFFFFFFF)
        self.worlds[self.idx] = build_world(self.worlds[self.idx]["id"], new_seed)
        self._draw()

    def _regen_all(self, _):
        print("Regenerating all 30 worlds...")
        self.worlds = generate_all_worlds(
            seeds=[random.randint(0, 0xFFFFFFFF) for _ in range(NUM_WORLDS)]
        )
        self._draw()

    def _save_current(self, _):
        os.makedirs(IMG_DIR, exist_ok=True)
        save_world_image(self.worlds[self.idx])
        print(f"  Saved world {self.worlds[self.idx]['id']} PNG")

    def _save_all(self, _):
        print("Saving all worlds...")
        save_worlds(self.worlds)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  GRIDWORLD LAB — Part 0")
    print(f"  {NUM_WORLDS} worlds · {GRID_SIZE}×{GRID_SIZE} · {int(BLOCK_PROB*100)}% block prob")
    print("=" * 50)

    # Load existing worlds if available, else generate fresh
    if os.path.exists(JSON_PATH):
        print(f"\nFound existing worlds at {JSON_PATH}")
        choice = input("  Load existing? [Y/n]: ").strip().lower()
        if choice != "n":
            worlds = load_worlds()
        else:
            print("\nGenerating 30 new worlds...")
            worlds = generate_all_worlds()
    else:
        print("\nGenerating 30 new worlds...")
        worlds = generate_all_worlds()

    print("\nLaunching viewer...")
    viewer = GridWorldViewer(worlds)
