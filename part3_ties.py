"""
Part 3 - The Effects of Ties
=============================
Compares two tie-breaking strategies for Repeated Forward A*:
  - Large-g: prefer cells with LARGER g-values (closer to goal)
  - Small-g: prefer cells with SMALLER g-values (closer to start)

Priority formula (single integer, as hinted in project):
  Large-g:  priority = C * f(s) - g(s) →  larger g wins ties
  Small-g:  priority = C * f(s) + g(s) →  smaller g wins ties
  where C = GRID_SIZE * GRID_SIZE = 2601 (larger than any g-value)

Requirements:
    pip install matplotlib numpy

Run:
    python part3_ties.py
"""

import json, os, random, heapq, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE  = 51
NUM_WORLDS = 30
BLOCK_PROB = 0.30
DIRS4      = [(-1,0),(1,0),(0,-1),(0,1)]
JSON_PATH  = os.path.join("gridworlds", "gridworlds_30.json")
INF        = float('inf') # means "no path found yet"
C          = GRID_SIZE * GRID_SIZE   # = 2601, tie-breaking constant

CMAP_GRID  = ListedColormap(["#e8eaf6", "#1a1a2e"])

# ═══════════════════════════════════════════════════════════════════════════════
# MAZE GENERATION (same as Part 0)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_maze(seed):
    rng  = random.Random(seed)
    rows = cols = GRID_SIZE
    grid    = np.ones((rows, cols), dtype=np.int8)
    visited = np.zeros((rows, cols), dtype=bool)
    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def unvisited_nbrs(r, c):
        d = DIRS4[:]; rng.shuffle(d)
        return [(r+dr,c+dc) for dr,dc in d
                if in_bounds(r+dr,c+dc) and not visited[r+dr][c+dc]]
    all_cells = [(r,c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)
    unvisited = set(all_cells)
    sr,sc = rng.randint(0,rows-1), rng.randint(0,cols-1)
    visited[sr][sc]=True; grid[sr][sc]=0; unvisited.discard((sr,sc))
    stack=[(sr,sc)]
    while unvisited:
        if not stack:
            nr,nc=next(iter(unvisited)); visited[nr][nc]=True; unvisited.discard((nr,nc))
            grid[nr][nc]=1 if rng.random()<BLOCK_PROB else 0
            if grid[nr][nc]==0: stack.append((nr,nc))
            continue
        r,c=stack[-1]; nbrs=unvisited_nbrs(r,c)
        if not nbrs: stack.pop()
        else:
            nr,nc=nbrs[0]; visited[nr][nc]=True; unvisited.discard((nr,nc))
            if rng.random()<BLOCK_PROB: grid[nr][nc]=1
            else: grid[nr][nc]=0; stack.append((nr,nc))
    return grid

def place_agent_target(grid, rng):
    free=list(zip(*np.where(grid==0)))
    if len(free)<2: return (0,0),(GRID_SIZE-1,GRID_SIZE-1)
    ai=rng.randint(0,len(free)-1)
    ti=rng.randint(0,len(free)-2)
    if ti>=ai: ti+=1
    return tuple(map(int,free[ai])),tuple(map(int,free[ti]))

def build_world(wid, seed):
    grid=generate_maze(seed)
    rng=random.Random(seed^0xDEADBEEF)
    agent,target=place_agent_target(grid,rng)
    return {"id":wid,"seed":seed,"grid":grid,"agent":agent,"target":target}

def load_or_generate_worlds():
    if os.path.exists(JSON_PATH):
        print(f"Loading worlds from {JSON_PATH}...")
        with open(JSON_PATH) as f: data=json.load(f)
        for d in data:
            d["grid"]=np.array(d["grid"],dtype=np.int8)
            d["agent"]=tuple(d["agent"]); d["target"]=tuple(d["target"])
        print(f"  Loaded {len(data)} worlds."); return data
    print("Generating 30 worlds...")
    worlds=[]
    for i in range(NUM_WORLDS):
        worlds.append(build_world(i+1,(i+1)*0x9E3779B9&0xFFFFFFFF))
    print("  Done."); return worlds


# ═══════════════════════════════════════════════════════════════════════════════
# CORE A* ENGINE  (large_g flag controls tie-breaking)
# ═══════════════════════════════════════════════════════════════════════════════
def manhattan(r1,c1,r2,c2): return abs(r1-r2)+abs(c1-c2) # Counts steps between two cells

class ForwardAStar:
    """
    Repeated Forward A* with selectable tie-breaking.

    large_g=True  → priority = C*f - g   (prefer LARGER g, i.e. closer to goal)
    large_g=False → priority = C*f + g   (prefer SMALLER g, i.e. closer to start)
    """
    def __init__(self, true_grid, agent, target, large_g=True): # Controls tie-breaking: True  = Large-g version, False = Small-g version
        self.true_grid = true_grid
        self.agent     = list(agent)
        self.target    = target
        self.large_g   = large_g
        self.rows = self.cols = GRID_SIZE

        self.known   = np.full((self.rows,self.cols),-1,dtype=np.int8)
        self.g       = np.full((self.rows,self.cols),INF)
        self.search  = np.zeros((self.rows,self.cols),dtype=np.int32)
        self.parent  = {}
        self.counter = 0

        # Pre-compute h (Manhattan to target) — never changes
        tr,tc = target
        self.h = np.array([[manhattan(r,c,tr,tc)
                            for c in range(self.cols)]
                           for r in range(self.rows)], dtype=np.float64)

        # Stats
        self.total_moves    = 0
        self.num_searches   = 0
        self.total_expanded = 0
        self.runtime        = 0.0

        # History for visualization
        self.history = []

        self._observe(tuple(self.agent))

    def _in_bounds(self,r,c): return 0<=r<self.rows and 0<=c<self.cols
    def _is_blocked(self,r,c): return self.known[r][c]==1

    def _observe(self,pos):
        r,c=pos; self.known[r][c]=0
        for dr,dc in DIRS4:
            nr,nc=r+dr,c+dc
            if self._in_bounds(nr,nc):
                self.known[nr][nc]=self.true_grid[nr][nc]
                
    "KEY Function : tie breaking"
    def _priority(self, f, g):
        """Single-integer priority encoding both f and tie-breaking."""
        if self.large_g:
            return int(C * f - g) # larger g 
        else:
            return int(C * f + g) # smaller g 

    def _compute_path(self):
        self.counter += 1
        self.num_searches += 1
        ctr = self.counter

        sr,sc = self.agent
        gr,gc = self.target

        # Lazy init start
        self.search[sr][sc]=ctr; self.g[sr][sc]=0
        # Lazy init goal
        if self.search[gr][gc]!=ctr:
            self.search[gr][gc]=ctr; self.g[gr][gc]=INF

        open_heap = []
        p0 = self._priority(self.h[sr][sc], 0) # ← uses Large-g or Small-g formula
        heapq.heappush(open_heap,(p0, sr, sc))

        closed    = set()
        expanded  = []

        while open_heap:
            p,r,c = heapq.heappop(open_heap)
            if (r,c) in closed: continue

            g_cur = self.g[r][c]
            f_cur = g_cur + self.h[r][c]

            # Termination: best node's f ≥ g(goal)
            if f_cur > self.g[gr][gc]: break

            closed.add((r,c))
            expanded.append((r,c))
            self.total_expanded += 1

            for dr,dc in DIRS4:
                nr,nc=r+dr,c+dc
                if not self._in_bounds(nr,nc): continue
                if self._is_blocked(nr,nc): continue
                if self.search[nr][nc]!=ctr:
                    self.search[nr][nc]=ctr; self.g[nr][nc]=INF
                new_g = g_cur+1
                if new_g < self.g[nr][nc]:
                    self.g[nr][nc]=new_g
                    self.parent[(nr,nc)]=(r,c)
                    prio=self._priority(new_g+self.h[nr][nc], new_g)
                    heapq.heappush(open_heap,(prio,nr,nc))

        if self.g[gr][gc]==INF: return None, expanded

        # Reconstruct path
        path=[]; cur=self.target
        while cur!=tuple(self.agent):
            path.append(cur); cur=self.parent.get(cur)
            if cur is None: return None, expanded
        path.append(tuple(self.agent)); path.reverse()
        return path, expanded
    #
    def run(self):
        trajectory=[tuple(self.agent)]
        t0=time.perf_counter() #start time

        while tuple(self.agent)!=self.target:
            path,expanded=self._compute_path()

            if path is None:
                self.runtime=time.perf_counter()-t0
                self.history.append({
                    "type":"search_failed","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":expanded,
                    "path":None,"trajectory":list(trajectory),
                    "search_num":self.num_searches,
                })
                return False, trajectory

            self.history.append({
                "type":"search","agent":tuple(self.agent),
                "known":self.known.copy(),"expanded":list(expanded),
                "path":list(path),"trajectory":list(trajectory),
                "search_num":self.num_searches,
            })

            for i in range(1,len(path)):
                nc_pos=path[i]
                self._observe(nc_pos)
                nr,nc=nc_pos
                if self._is_blocked(nr,nc): break
                self.agent=list(nc_pos)
                self.total_moves+=1
                trajectory.append(tuple(self.agent))
                self.history.append({
                    "type":"move","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":[],
                    "path":path[i:],"trajectory":list(trajectory),
                    "search_num":self.num_searches,
                })
                if tuple(self.agent)==self.target:
                    self.runtime=time.perf_counter()-t0
                    return True, trajectory
                blocked_ahead=any(self._is_blocked(path[j][0],path[j][1])
                                  for j in range(i+1,len(path)))
                if blocked_ahead: break

        self.runtime=time.perf_counter()-t0 #stop timer.
        return True, trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BOTH VERSIONS ON ONE WORLD
# ═══════════════════════════════════════════════════════════════════════════════
def run_both(world):
    print(f"\n  Running Large-g A* on World #{world['id']}...")
    lg = ForwardAStar(world["grid"], world["agent"], world["target"], large_g=True)
    lg_success, lg_traj = lg.run()

    print(f"  Running Small-g A* on World #{world['id']}...")
    sg = ForwardAStar(world["grid"], world["agent"], world["target"], large_g=False)
    sg_success, sg_traj = sg.run()

    return {
        "large_g": {"solver":lg,"success":lg_success,"trajectory":lg_traj},
        "small_g": {"solver":sg,"success":sg_success,"trajectory":sg_traj},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE ACROSS ALL 30 WORLDS
# ═══════════════════════════════════════════════════════════════════════════════
def run_comparison_all(worlds):
    print("\n" + "="*72)
    print("  COMPARISON: Large-g vs Small-g across all 30 worlds")
    print("="*72)
    print(f"  {'World':>5} | {'LG-Exp':>8} | {'SG-Exp':>8} | "
          f"{'LG-Moves':>8} | {'SG-Moves':>8} | "
          f"{'LG-ms':>7} | {'SG-ms':>7} | {'Winner':>8}")
    print("  " + "-"*70)

    results = []
    lg_exp_total=sg_exp_total=lg_mv_total=sg_mv_total=0
    lg_rt_total=sg_rt_total=0.0
    lg_wins=sg_wins=ties=0

    for w in worlds:
        lg=ForwardAStar(w["grid"],w["agent"],w["target"],large_g=True)
        lg_ok,_=lg.run()
        sg=ForwardAStar(w["grid"],w["agent"],w["target"],large_g=False)
        sg_ok,_=sg.run()

        le,se=lg.total_expanded,sg.total_expanded
        lm,sm=lg.total_moves,sg.total_moves
        lr,sr_=lg.runtime*1000,sg.runtime*1000

        lg_exp_total+=le; sg_exp_total+=se
        lg_mv_total+=lm;  sg_mv_total+=sm
        lg_rt_total+=lr;  sg_rt_total+=sr_

        if le<se:   winner="Large-g ✅"; lg_wins+=1
        elif se<le: winner="Small-g"; sg_wins+=1
        else:       winner="Tie"; ties+=1

        results.append({"id":w["id"],"lg_exp":le,"sg_exp":se,
                        "lg_moves":lm,"sg_moves":sm,
                        "lg_rt":lr,"sg_rt":sr_,"winner":winner})

        print(f"  {w['id']:>5} | {le:>8,} | {se:>8,} | "
              f"{lm:>8,} | {sm:>8,} | "
              f"{lr:>7.1f} | {sr_:>7.1f} | {winner}")

    n=len(worlds)
    print("  " + "="*70)
    print(f"  {'AVG':>5} | {lg_exp_total//n:>8,} | {sg_exp_total//n:>8,} | "
          f"{lg_mv_total//n:>8,} | {sg_mv_total//n:>8,} | "
          f"{lg_rt_total/n:>7.1f} | {sg_rt_total/n:>7.1f} |")
    print(f"\n  Large-g wins: {lg_wins}/30   Small-g wins: {sg_wins}/30   Ties: {ties}/30")
    print(f"  Large-g avg expanded: {lg_exp_total//n:,}  vs  "
          f"Small-g avg expanded: {sg_exp_total//n:,}")
    pct = (sg_exp_total-lg_exp_total)/sg_exp_total*100 if sg_exp_total>0 else 0
    print(f"  Large-g expands {pct:.1f}% FEWER cells on average")
    print("="*72)

    plot_comparison_chart(results)
    return results


def plot_comparison_chart(results):
    """Bar chart comparing expanded cells for all 30 worlds."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d0d18")
    fig.canvas.manager.set_window_title("Part 3 — Large-g vs Small-g Comparison")

    worlds_ids = [r["id"] for r in results]
    lg_exp     = [r["lg_exp"]   for r in results]
    sg_exp     = [r["sg_exp"]   for r in results]
    lg_rt      = [r["lg_rt"]    for r in results]
    sg_rt      = [r["sg_rt"]    for r in results]
    x          = np.arange(len(results))
    w          = 0.38

    # ── Top: Expanded cells ───────────────────────────────────────────────────
    ax1 = axes[0]; ax1.set_facecolor("#0d0d18")
    bars1 = ax1.bar(x-w/2, lg_exp, w, label="Large-g", color="#00e5b0", alpha=0.85)
    bars2 = ax1.bar(x+w/2, sg_exp, w, label="Small-g", color="#ff5370", alpha=0.85)
    ax1.set_title("Cells Expanded: Large-g vs Small-g",
                  color="#e0e0f0", fontsize=11)
    ax1.set_ylabel("Expanded Cells", color="#8888aa")
    ax1.set_xticks(x); ax1.set_xticklabels(worlds_ids, fontsize=7, color="#8888aa")
    ax1.tick_params(colors="#8888aa")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
               fontsize=9)
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")
    ax1.yaxis.label.set_color("#8888aa")

    # Avg lines
    avg_lg = sum(lg_exp)/len(lg_exp)
    avg_sg = sum(sg_exp)/len(sg_exp)
    ax1.axhline(avg_lg, color="#00e5b0", linewidth=1.5,
                linestyle="--", alpha=0.6, label=f"LG avg={avg_lg:.0f}")
    ax1.axhline(avg_sg, color="#ff5370", linewidth=1.5,
                linestyle="--", alpha=0.6, label=f"SG avg={avg_sg:.0f}")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333",
               labelcolor="white", fontsize=8)

    # ── Bottom: Runtime ───────────────────────────────────────────────────────
    ax2 = axes[1]; ax2.set_facecolor("#0d0d18")
    ax2.bar(x-w/2, lg_rt, w, label="Large-g (ms)", color="#7c6af7", alpha=0.85)
    ax2.bar(x+w/2, sg_rt, w, label="Small-g (ms)", color="#ffab40", alpha=0.85)
    ax2.set_title("Runtime (ms): Large-g vs Small-g",
                  color="#e0e0f0", fontsize=11)
    ax2.set_ylabel("Runtime (ms)", color="#8888aa")
    ax2.set_xlabel("World #",      color="#8888aa")
    ax2.set_xticks(x); ax2.set_xticklabels(worlds_ids, fontsize=7, color="#8888aa")
    ax2.tick_params(colors="#8888aa")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333",
               labelcolor="white", fontsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")

    fig.patch.set_facecolor("#0d0d18")
    plt.tight_layout(pad=2)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP VISUALIZER (side-by-side Large-g vs Small-g)
# ═══════════════════════════════════════════════════════════════════════════════
class TieBreakVisualizer:
    """
    Shows Large-g and Small-g side-by-side at each step.
    Highlights expanded cells in different colors for easy comparison.
    """
    def __init__(self, world, results):
        self.world   = world
        self.results = results
        self.lg_hist = results["large_g"]["solver"].history
        self.sg_hist = results["small_g"]["solver"].history
        self.step_lg = 0
        self.step_sg = 0
        self.sync    = True   # sync both panels to same step index
        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 8), facecolor="#0d0d18")
        self.fig.canvas.manager.set_window_title(
            f"Part 3 — Tie-Breaking Comparison — World #{self.world['id']}")

        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               height_ratios=[10,1],
                               hspace=0.25, wspace=0.12)

        self.ax_true = self.fig.add_subplot(gs[0,0])
        self.ax_lg   = self.fig.add_subplot(gs[0,1])
        self.ax_sg   = self.fig.add_subplot(gs[0,2])

        for ax in [self.ax_true,self.ax_lg,self.ax_sg]:
            ax.set_facecolor("#0d0d18")
            ax.set_xticks([]); ax.set_yticks([])

        # Buttons row
        btn_specs = [
            (0.02,  "|◀",  self._goto_start),
            (0.10,  "◀",   lambda e: self._step(-1)),
            (0.18,  "▶",   lambda e: self._step(+1)),
            (0.26,  "▶|",  self._goto_end),
            (0.38,  "AUTO",self._autoplay),
        ]
        self.buttons = []
        for x, label, cb in btn_specs:
            ax_b = self.fig.add_axes([x, 0.02, 0.07, 0.05])
            b = Button(ax_b, label, color="#1a1a2e", hovercolor="#2a2a40")
            b.label.set_color("#e0e0f0")
            b.label.set_family("monospace")
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            self.buttons.append(b)

        self.max_steps = max(len(self.lg_hist), len(self.sg_hist))
        self._draw()
        plt.show()

    def _step(self, d):
        self.step_lg = max(0, min(len(self.lg_hist)-1, self.step_lg+d))
        self.step_sg = max(0, min(len(self.sg_hist)-1, self.step_sg+d))
        self._draw()

    def _goto_start(self, e): self.step_lg=0; self.step_sg=0; self._draw()
    def _goto_end(self, e):
        self.step_lg=len(self.lg_hist)-1
        self.step_sg=len(self.sg_hist)-1
        self._draw()

    def _autoplay(self, event):
        steps = max(len(self.lg_hist), len(self.sg_hist))
        for i in range(steps):
            self.step_lg = min(i, len(self.lg_hist)-1)
            self.step_sg = min(i, len(self.sg_hist)-1)
            self._draw()
            plt.pause(0.08)

    def _draw_panel(self, ax, hist, step, title, color_exp, color_path):
        h = hist[min(step, len(hist)-1)]
        w = self.world
        tr,tc = w["target"]

        ax.cla(); ax.set_facecolor("#0d0d18")
        known   = h["known"]
        display = np.where(known==1,1,0).astype(np.float32)
        ax.imshow(display, cmap=CMAP_GRID, vmin=0, vmax=1,
                  origin="upper", interpolation="nearest")

        # Unknown cells
        unk = np.zeros((*display.shape,4))
        unk[known==-1] = [0.7,0.7,0.8,0.25]
        ax.imshow(unk, origin="upper", interpolation="nearest", zorder=1)

        # Expanded cells
        if h["expanded"]:
            exp_ov = np.zeros((*display.shape,4))
            for er,ec in h["expanded"]:
                rgb = (*[x/255 for x in
                         (int(color_exp[1:3],16),
                          int(color_exp[3:5],16),
                          int(color_exp[5:7],16))], 0.5)
                exp_ov[er,ec] = rgb
            ax.imshow(exp_ov, origin="upper", interpolation="nearest", zorder=2)

        # Path
        if h["path"] and len(h["path"])>1:
            pr=[p[0] for p in h["path"]]
            pc=[p[1] for p in h["path"]]
            ax.plot(pc,pr,"-",color=color_path,linewidth=2.5,zorder=4,alpha=0.9)

        # Trajectory
        traj=h["trajectory"]
        if len(traj)>1:
            ax.plot([p[1] for p in traj],[p[0] for p in traj],
                    "-",color="#ffffff",linewidth=1.2,alpha=0.5,zorder=3)

        # Agent & Target
        ar,ac=h["agent"]
        ax.plot(ac,ar,"o",color="#00e5b0",markersize=9,
                markeredgecolor="white",markeredgewidth=1.5,zorder=6)
        ax.text(ac,ar,"A",color="white",fontsize=5,
                ha="center",va="center",fontweight="bold",zorder=7)
        ax.plot(tc,tr,"o",color="#ff5370",markersize=9,
                markeredgecolor="white",markeredgewidth=1.5,zorder=6)
        ax.text(tc,tr,"T",color="white",fontsize=5,
                ha="center",va="center",fontweight="bold",zorder=7)

        stype = h["type"]
        snum  = h.get("search_num","")
        label = {"search":f"Search #{snum} | {len(h['expanded'])} expanded",
                 "move":"Moving",
                 "search_failed":"❌ No path"}.get(stype,stype)
        ax.set_title(f"{title}\n{label}",
                     color=color_path,fontsize=9,pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    def _draw(self):
        w  = self.world
        tr,tc = w["target"]

        # True grid (left panel)
        self.ax_true.cla(); self.ax_true.set_facecolor("#0d0d18")
        self.ax_true.imshow(w["grid"],cmap=CMAP_GRID,vmin=0,vmax=1,
                            origin="upper",interpolation="nearest")
        ar0,ac0 = w["agent"]
        self.ax_true.plot(ac0,ar0,"o",color="#00e5b0",markersize=9,
                          markeredgecolor="white",markeredgewidth=1.5,zorder=5)
        self.ax_true.text(ac0,ar0,"A",color="white",fontsize=5,
                          ha="center",va="center",fontweight="bold",zorder=6)
        self.ax_true.plot(tc,tr,"o",color="#ff5370",markersize=9,
                          markeredgecolor="white",markeredgewidth=1.5,zorder=5)
        self.ax_true.text(tc,tr,"T",color="white",fontsize=5,
                          ha="center",va="center",fontweight="bold",zorder=6)
        self.ax_true.set_title("TRUE GRIDWORLD",color="#7c6af7",
                               fontsize=10,pad=6)

        # Large-g panel (middle)
        self._draw_panel(self.ax_lg, self.lg_hist, self.step_lg,
                         "LARGE-g  (prefer closer to goal)",
                         "#00e5b044", "#00e5b0")

        # Small-g panel (right)
        self._draw_panel(self.ax_sg, self.sg_hist, self.step_sg,
                         "SMALL-g  (prefer closer to start)",
                         "#ff537044", "#ff5370")

        # Legend
        patches = [
            mpatches.Patch(color="#00e5b0",alpha=0.5,label="Large-g expanded"),
            mpatches.Patch(color="#ff5370",alpha=0.5,label="Small-g expanded"),
            mpatches.Patch(color="#00e5b0",label="Large-g path"),
            mpatches.Patch(color="#ff5370",label="Small-g path"),
            mpatches.Patch(color="#ffffff",alpha=0.5,label="Trajectory"),
        ]
        self.ax_sg.legend(handles=patches,loc="lower left",fontsize=6,
                          facecolor="#1a1a2e",edgecolor="#333",
                          labelcolor="white",framealpha=0.9)

        # Stats footer
        lg_s=self.results["large_g"]["solver"]
        sg_s=self.results["small_g"]["solver"]
        footer=(f"Step {self.step_lg+1}/{len(self.lg_hist)}  |  "
                f"Large-g: {lg_s.total_expanded:,} expanded, "
                f"{lg_s.total_moves} moves, {lg_s.runtime*1000:.1f}ms  ‖  "
                f"Small-g: {sg_s.total_expanded:,} expanded, "
                f"{sg_s.total_moves} moves, {sg_s.runtime*1000:.1f}ms")
        self.fig.texts.clear()
        self.fig.text(0.5,0.005,footer,ha="center",color="#6668a0",
                      fontsize=8)
        self.fig.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    worlds = load_or_generate_worlds()

    print("\n" + "="*50)
    print("  PART 3 — THE EFFECTS OF TIES")
    print("="*50)
    print("\n  Options:")
    print("  1. Visualize step-by-step on one world")
    print("  2. Run comparison on ALL 30 worlds + chart")
    print("  3. Both")

    choice = input("\n  Enter choice [1/2/3]: ").strip()

    if choice in ("1","3"):
        print("\n  Enter world number (1-30) or press Enter for world 1:")
        idx_in = input("  World: ").strip()
        idx = (int(idx_in)-1) if idx_in.isdigit() else 0
        idx = max(0, min(NUM_WORLDS-1, idx))
        world = worlds[idx]
        print(f"\n  Running both versions on World #{world['id']}...")
        results = run_both(world)
        lg = results["large_g"]["solver"]
        sg = results["small_g"]["solver"]
        print(f"\n  ┌──────────────┬────────────┬────────────┐")
        print(f"  │              │  Large-g   │  Small-g   │")
        print(f"  ├──────────────┼────────────┼────────────┤")
        print(f"  │ Expanded     │ {lg.total_expanded:>10,} │ {sg.total_expanded:>10,} │")
        print(f"  │ Moves        │ {lg.total_moves:>10,} │ {sg.total_moves:>10,} │")
        print(f"  │ Searches     │ {lg.num_searches:>10,} │ {sg.num_searches:>10,} │")
        print(f"  │ Runtime (ms) │ {lg.runtime*1000:>10.2f} │ {sg.runtime*1000:>10.2f} │")
        print(f"  └──────────────┴────────────┴────────────┘")
        print(f"\n  Launching side-by-side visualizer...")
        TieBreakVisualizer(world, results)

    if choice in ("2","3"):
        run_comparison_all(worlds)

    print("\n  Done!")
