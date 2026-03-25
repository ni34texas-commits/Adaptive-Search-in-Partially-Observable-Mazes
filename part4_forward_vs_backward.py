"""
Part 4 - Forward A* vs Backward A*
=====================================
Compares Repeated Forward A* vs Repeated Backward A*.

Key difference:
  Forward:  searches agent → target  (start=agent, goal=target)
  Backward: searches target → agent  (start=target, goal=agent)

Both use:
  - Large-g tie-breaking: priority = C * f(s) - g(s)
  - Manhattan distance heuristic
  - Lazy initialization (counter stamps)
  - Explicit closed list

Requirements:
    pip install matplotlib numpy

Run:
    python part4_forward_vs_backward.py
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
INF        = float('inf')
C          = GRID_SIZE * GRID_SIZE   # tie-breaking constant = 2601
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
            nr,nc=next(iter(unvisited)); visited[nr][nc]=True
            unvisited.discard((nr,nc))
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
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def manhattan(r1,c1,r2,c2):
    return abs(r1-r2)+abs(c1-c2)

def priority(f, g):
    """Large-g tie-breaking: bigger g = higher priority."""
    return int(C * f - g)

# ═══════════════════════════════════════════════════════════════════════════════
# REPEATED FORWARD A*
# Searches: agent (start) → target (goal)
# h(s) = Manhattan distance from s to TARGET (fixed goal, never changes)
# ═══════════════════════════════════════════════════════════════════════════════
class ForwardAStar:
    def __init__(self, true_grid, agent, target):
        self.true_grid = true_grid
        self.agent     = list(agent)
        self.target    = target
        self.rows = self.cols = GRID_SIZE

        self.known   = np.full((self.rows,self.cols),-1,dtype=np.int8)
        self.g       = np.full((self.rows,self.cols),INF)
        self.search  = np.zeros((self.rows,self.cols),dtype=np.int32)
        self.parent  = {}
        self.counter = 0

        # h(s) = Manhattan to TARGET — computed once, never changes
        tr,tc = target
        self.h = np.array([[manhattan(r,c,tr,tc)
                            for c in range(self.cols)]
                           for r in range(self.rows)], dtype=np.float64)

        self.total_moves    = 0
        self.num_searches   = 0
        self.total_expanded = 0
        self.runtime        = 0.0
        self.history        = []

        self._observe(tuple(self.agent))

    def _in_bounds(self,r,c): return 0<=r<self.rows and 0<=c<self.cols
    def _is_blocked(self,r,c): return self.known[r][c]==1

    def _observe(self, pos):
        r,c=pos; self.known[r][c]=0
        for dr,dc in DIRS4:
            nr,nc=r+dr,c+dc
            if self._in_bounds(nr,nc):
                self.known[nr][nc]=self.true_grid[nr][nc]

    def _compute_path(self):
        """A* from agent → target."""
        self.counter+=1; self.num_searches+=1
        ctr=self.counter
        sr,sc=self.agent
        gr,gc=self.target

        # Lazy init
        self.search[sr][sc]=ctr; self.g[sr][sc]=0
        if self.search[gr][gc]!=ctr:
            self.search[gr][gc]=ctr; self.g[gr][gc]=INF

        heap=[]; heapq.heappush(heap,(priority(self.h[sr][sc],0),sr,sc))
        closed=set(); expanded=[]

        while heap:
            p,r,c=heapq.heappop(heap)
            if (r,c) in closed: continue
            g_cur=self.g[r][c]
            if g_cur+self.h[r][c]>self.g[gr][gc]: break
            closed.add((r,c)); expanded.append((r,c))
            self.total_expanded+=1

            for dr,dc in DIRS4:
                nr,nc=r+dr,c+dc
                if not self._in_bounds(nr,nc) or self._is_blocked(nr,nc): continue
                if self.search[nr][nc]!=ctr:
                    self.search[nr][nc]=ctr; self.g[nr][nc]=INF
                new_g=g_cur+1
                if new_g<self.g[nr][nc]:
                    self.g[nr][nc]=new_g
                    self.parent[(nr,nc)]=(r,c)
                    heapq.heappush(heap,(priority(new_g+self.h[nr][nc],new_g),nr,nc))

        if self.g[gr][gc]==INF: return None, expanded

        # Reconstruct path: follow parents from goal back to agent
        path=[]; cur=self.target
        start=tuple(self.agent)
        while cur!=start:
            path.append(cur)
            cur=self.parent.get(cur)
            if cur is None: return None, expanded
        path.append(start); path.reverse()
        return path, expanded

    def run(self):
        trajectory=[tuple(self.agent)]
        t0=time.perf_counter()

        while tuple(self.agent)!=self.target:
            path,expanded=self._compute_path()
            if path is None:
                self.runtime=time.perf_counter()-t0
                self.history.append({
                    "type":"search_failed","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":expanded,
                    "path":None,"trajectory":list(trajectory),
                    "search_num":self.num_searches,"direction":"forward"
                })
                return False, trajectory

            self.history.append({
                "type":"search","agent":tuple(self.agent),
                "known":self.known.copy(),"expanded":list(expanded),
                "path":list(path),"trajectory":list(trajectory),
                "search_num":self.num_searches,"direction":"forward"
            })

            for i in range(1,len(path)):
                nxt=path[i]; nr,nc=nxt
                self._observe(nxt)
                if self._is_blocked(nr,nc): break
                self.agent=list(nxt)
                self.total_moves+=1
                trajectory.append(tuple(self.agent))
                self.history.append({
                    "type":"move","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":[],
                    "path":path[i:],"trajectory":list(trajectory),
                    "search_num":self.num_searches,"direction":"forward"
                })
                if tuple(self.agent)==self.target:
                    self.runtime=time.perf_counter()-t0
                    return True, trajectory
                if any(self._is_blocked(path[j][0],path[j][1])
                       for j in range(i+1,len(path))): break

        self.runtime=time.perf_counter()-t0
        return True, trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# REPEATED BACKWARD A*
# Searches: target (start) → agent (goal)
# h(s) = Manhattan distance from s to AGENT
#   → NOTE: agent position changes every replanning step,
#     so h-values are recomputed at the start of each A* search.
# The path returned is target→agent, then REVERSED to get agent→target.
# ═══════════════════════════════════════════════════════════════════════════════
class BackwardAStar:
    def __init__(self, true_grid, agent, target):
        self.true_grid = true_grid
        self.agent     = list(agent)
        self.target    = target
        self.rows = self.cols = GRID_SIZE

        # Agent's knowledge map
        self.known   = np.full((self.rows,self.cols),-1,dtype=np.int8)
        self.g       = np.full((self.rows,self.cols),INF)
        self.search  = np.zeros((self.rows,self.cols),dtype=np.int32)
        self.parent  = {}
        self.counter = 0

        # h-values recomputed each search (because agent/goal changes)
        self.h = np.zeros((self.rows,self.cols),dtype=np.float64)

        self.total_moves    = 0
        self.num_searches   = 0
        self.total_expanded = 0
        self.runtime        = 0.0
        self.history        = []

        self._observe(tuple(self.agent))

    def _in_bounds(self,r,c): return 0<=r<self.rows and 0<=c<self.cols
    def _is_blocked(self,r,c): return self.known[r][c]==1

    def _observe(self, pos):
        """Agent observes its 4 neighbors."""
        r,c=pos; self.known[r][c]=0
        for dr,dc in DIRS4:
            nr,nc=r+dr,c+dc
            if self._in_bounds(nr,nc):
                self.known[nr][nc]=self.true_grid[nr][nc]

    def _recompute_h(self):
        """
        Recompute h-values = Manhattan distance to AGENT (current goal).
        Must be called at start of each search because agent has moved.
        """
        ar,ac=self.agent
        for r in range(self.rows):
            for c in range(self.cols):
                self.h[r][c]=manhattan(r,c,ar,ac)

    def _compute_path(self):
        """
        A* from TARGET → AGENT.
        Start = target (fixed), Goal = agent (changes each search).
        Returns path from AGENT → TARGET (reversed after reconstruction).
        """
        # Recompute h because agent (= goal) has moved
        self._recompute_h()

        self.counter+=1; self.num_searches+=1
        ctr=self.counter

        # In backward search: start=target, goal=agent
        sr,sc=self.target
        gr,gc=tuple(self.agent)

        # Lazy init
        self.search[sr][sc]=ctr; self.g[sr][sc]=0
        if self.search[gr][gc]!=ctr:
            self.search[gr][gc]=ctr; self.g[gr][gc]=INF

        heap=[]; heapq.heappush(heap,(priority(self.h[sr][sc],0),sr,sc))
        closed=set(); expanded=[]

        while heap:
            p,r,c=heapq.heappop(heap)
            if (r,c) in closed: continue
            g_cur=self.g[r][c]
            if g_cur+self.h[r][c]>self.g[gr][gc]: break
            closed.add((r,c)); expanded.append((r,c))
            self.total_expanded+=1

            for dr,dc in DIRS4:
                nr,nc=r+dr,c+dc
                if not self._in_bounds(nr,nc) or self._is_blocked(nr,nc): continue
                if self.search[nr][nc]!=ctr:
                    self.search[nr][nc]=ctr; self.g[nr][nc]=INF
                new_g=g_cur+1
                if new_g<self.g[nr][nc]:
                    self.g[nr][nc]=new_g
                    self.parent[(nr,nc)]=(r,c)
                    heapq.heappush(heap,(priority(new_g+self.h[nr][nc],new_g),nr,nc))

        if self.g[gr][gc]==INF: return None, expanded

        # Reconstruct: follow parents from AGENT back to TARGET
        # (backward search stored parents in target→agent direction)
        path_backward=[]
        cur=(gr,gc)  # start at agent
        goal_pos=(sr,sc)  # end at target
        while cur!=goal_pos:
            path_backward.append(cur)
            cur=self.parent.get(cur)
            if cur is None: return None, expanded
        path_backward.append(goal_pos)

        # path_backward is: agent → ... → target
        # This is already the correct direction for the agent to follow!
        return path_backward, expanded

    def run(self):
        trajectory=[tuple(self.agent)]
        t0=time.perf_counter()

        while tuple(self.agent)!=self.target:
            path,expanded=self._compute_path()
            if path is None:
                self.runtime=time.perf_counter()-t0
                self.history.append({
                    "type":"search_failed","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":expanded,
                    "path":None,"trajectory":list(trajectory),
                    "search_num":self.num_searches,"direction":"backward"
                })
                return False, trajectory

            self.history.append({
                "type":"search","agent":tuple(self.agent),
                "known":self.known.copy(),"expanded":list(expanded),
                "path":list(path),"trajectory":list(trajectory),
                "search_num":self.num_searches,"direction":"backward"
            })

            for i in range(1,len(path)):
                nxt=path[i]; nr,nc=nxt
                self._observe(nxt)
                if self._is_blocked(nr,nc): break
                self.agent=list(nxt)
                self.total_moves+=1
                trajectory.append(tuple(self.agent))
                self.history.append({
                    "type":"move","agent":tuple(self.agent),
                    "known":self.known.copy(),"expanded":[],
                    "path":path[i:],"trajectory":list(trajectory),
                    "search_num":self.num_searches,"direction":"backward"
                })
                if tuple(self.agent)==self.target:
                    self.runtime=time.perf_counter()-t0
                    return True, trajectory
                if any(self._is_blocked(path[j][0],path[j][1])
                       for j in range(i+1,len(path))): break

        self.runtime=time.perf_counter()-t0
        return True, trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BOTH ON ONE WORLD
# ═══════════════════════════════════════════════════════════════════════════════
def run_both(world):
    print(f"\n  Running Forward A* on World #{world['id']}...")
    fwd=ForwardAStar(world["grid"],world["agent"],world["target"])
    fwd_ok,fwd_traj=fwd.run()

    print(f"  Running Backward A* on World #{world['id']}...")
    bwd=BackwardAStar(world["grid"],world["agent"],world["target"])
    bwd_ok,bwd_traj=bwd.run()

    return {
        "forward":  {"solver":fwd,"success":fwd_ok,"trajectory":fwd_traj},
        "backward": {"solver":bwd,"success":bwd_ok,"trajectory":bwd_traj},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE ACROSS ALL 30 WORLDS
# ═══════════════════════════════════════════════════════════════════════════════
def run_comparison_all(worlds):
    print("\n"+"="*76)
    print("  COMPARISON: Forward A* vs Backward A* across all 30 worlds")
    print("="*76)
    print(f"  {'World':>5} | {'FWD-Exp':>8} | {'BWD-Exp':>8} | "
          f"{'FWD-Mv':>7} | {'BWD-Mv':>7} | "
          f"{'FWD-ms':>7} | {'BWD-ms':>7} | {'Winner':>10}")
    print("  "+"-"*74)

    results=[]
    fe_tot=be_tot=fm_tot=bm_tot=0
    fr_tot=br_tot=0.0
    fw_wins=bw_wins=ties=0

    for w in worlds:
        fwd=ForwardAStar(w["grid"],w["agent"],w["target"])
        fwd_ok,_=fwd.run()
        bwd=BackwardAStar(w["grid"],w["agent"],w["target"])
        bwd_ok,_=bwd.run()

        fe,be=fwd.total_expanded,bwd.total_expanded
        fm,bm=fwd.total_moves,bwd.total_moves
        fr,br=fwd.runtime*1000,bwd.runtime*1000

        fe_tot+=fe; be_tot+=be
        fm_tot+=fm; bm_tot+=bm
        fr_tot+=fr; br_tot+=br

        if fe<be:   winner="Forward ✅"; fw_wins+=1
        elif be<fe: winner="Backward ✅"; bw_wins+=1
        else:       winner="Tie"; ties+=1

        results.append({"id":w["id"],
                        "fe":fe,"be":be,"fm":fm,"bm":bm,
                        "fr":fr,"br":br,"winner":winner})

        print(f"  {w['id']:>5} | {fe:>8,} | {be:>8,} | "
              f"{fm:>7,} | {bm:>7,} | "
              f"{fr:>7.1f} | {br:>7.1f} | {winner}")

    n=len(worlds)
    print("  "+"="*74)
    print(f"  {'AVG':>5} | {fe_tot//n:>8,} | {be_tot//n:>8,} | "
          f"{fm_tot//n:>7,} | {bm_tot//n:>7,} | "
          f"{fr_tot/n:>7.1f} | {br_tot/n:>7.1f} |")
    print(f"\n  Forward wins:  {fw_wins}/30")
    print(f"  Backward wins: {bw_wins}/30")
    print(f"  Ties:          {ties}/30")
    diff=be_tot-fe_tot
    pct=diff/be_tot*100 if be_tot>0 else 0
    print(f"\n  Forward  avg expanded: {fe_tot//n:,}")
    print(f"  Backward avg expanded: {be_tot//n:,}")
    if diff>0:
        print(f"  → Forward expands {pct:.1f}% FEWER cells on average")
    else:
        print(f"  → Backward expands {-pct:.1f}% FEWER cells on average")
    print("="*76)

    _plot_comparison(results)
    return results


def _plot_comparison(results):
    fig,axes=plt.subplots(2,1,figsize=(14,8),facecolor="#0d0d18")
    fig.canvas.manager.set_window_title("Part 4 — Forward vs Backward A* Comparison")

    ids=[r["id"] for r in results]
    fe=[r["fe"] for r in results]
    be=[r["be"] for r in results]
    fr=[r["fr"] for r in results]
    br=[r["br"] for r in results]
    x=np.arange(len(results)); w=0.38

    # Expanded cells
    ax1=axes[0]; ax1.set_facecolor("#0d0d18")
    ax1.bar(x-w/2,fe,w,label="Forward A*", color="#00e5b0",alpha=0.85)
    ax1.bar(x+w/2,be,w,label="Backward A*",color="#7c6af7",alpha=0.85)
    avg_f=sum(fe)/len(fe); avg_b=sum(be)/len(be)
    ax1.axhline(avg_f,color="#00e5b0",linewidth=1.5,linestyle="--",
                alpha=0.7,label=f"FWD avg={avg_f:.0f}")
    ax1.axhline(avg_b,color="#7c6af7",linewidth=1.5,linestyle="--",
                alpha=0.7,label=f"BWD avg={avg_b:.0f}")
    ax1.set_title("Expanded Cells: Forward vs Backward A*",
                  color="#e0e0f0",fontsize=11)
    ax1.set_ylabel("Expanded Cells",color="#8888aa")
    ax1.set_xticks(x); ax1.set_xticklabels(ids,fontsize=7,color="#8888aa")
    ax1.tick_params(colors="#8888aa")
    ax1.legend(facecolor="#1a1a2e",edgecolor="#333",
               labelcolor="white",fontsize=9)
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")

    # Runtime
    ax2=axes[1]; ax2.set_facecolor("#0d0d18")
    ax2.bar(x-w/2,fr,w,label="Forward A* (ms)", color="#00e5b0",alpha=0.85)
    ax2.bar(x+w/2,br,w,label="Backward A* (ms)",color="#7c6af7",alpha=0.85)
    ax2.set_title("Runtime (ms): Forward vs Backward A*",
                  color="#e0e0f0",fontsize=11)
    ax2.set_ylabel("Runtime (ms)",color="#8888aa")
    ax2.set_xlabel("World #",     color="#8888aa")
    ax2.set_xticks(x); ax2.set_xticklabels(ids,fontsize=7,color="#8888aa")
    ax2.tick_params(colors="#8888aa")
    ax2.legend(facecolor="#1a1a2e",edgecolor="#333",
               labelcolor="white",fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")

    fig.patch.set_facecolor("#0d0d18")
    plt.tight_layout(pad=2)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE STEP VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════
class FwdBwdVisualizer:
    """
    Three-panel visualizer:
      Left:   True gridworld
      Middle: Forward A* knowledge + search
      Right:  Backward A* knowledge + search
    """
    def __init__(self, world, results):
        self.world   = world
        self.results = results
        self.fwd_hist=results["forward"]["solver"].history
        self.bwd_hist=results["backward"]["solver"].history
        self.step=0
        self._build_ui()

    def _build_ui(self):
        self.fig=plt.figure(figsize=(16,8),facecolor="#0d0d18")
        self.fig.canvas.manager.set_window_title(
            f"Part 4 — Forward vs Backward A* — World #{self.world['id']}")

        gs=gridspec.GridSpec(1,3,figure=self.fig,wspace=0.1)
        self.ax_true=self.fig.add_subplot(gs[0,0])
        self.ax_fwd =self.fig.add_subplot(gs[0,1])
        self.ax_bwd =self.fig.add_subplot(gs[0,2])

        for ax in [self.ax_true,self.ax_fwd,self.ax_bwd]:
            ax.set_facecolor("#0d0d18")
            ax.set_xticks([]); ax.set_yticks([])

        # Button row
        specs=[
            (0.02,"|◀",self._goto_start),
            (0.10,"◀", lambda e:self._step(-1)),
            (0.18,"▶", lambda e:self._step(+1)),
            (0.26,"▶|",self._goto_end),
            (0.38,"AUTO",self._autoplay),
        ]
        self.btns=[]
        for xp,label,cb in specs:
            ax_b=self.fig.add_axes([xp,0.01,0.07,0.05])
            b=Button(ax_b,label,color="#1a1a2e",hovercolor="#2a2a40")
            b.label.set_color("#e0e0f0"); b.label.set_fontsize(9)
            b.on_clicked(cb); self.btns.append(b)

        self._draw()
        plt.show()

    def _step(self,d):
        mx=max(len(self.fwd_hist),len(self.bwd_hist))-1
        self.step=max(0,min(mx,self.step+d)); self._draw()
    def _goto_start(self,e): self.step=0; self._draw()
    def _goto_end(self,e):
        self.step=max(len(self.fwd_hist),len(self.bwd_hist))-1; self._draw()
    def _autoplay(self,e):
        mx=max(len(self.fwd_hist),len(self.bwd_hist))
        for i in range(mx):
            self.step=i; self._draw(); plt.pause(0.08)

    def _draw_panel(self, ax, hist, step, title, exp_color, path_color):
        h=hist[min(step,len(hist)-1)]
        w=self.world; tr,tc=w["target"]
        ax.cla(); ax.set_facecolor("#0d0d18")

        known=h["known"]
        display=np.where(known==1,1,0).astype(np.float32)
        ax.imshow(display,cmap=CMAP_GRID,vmin=0,vmax=1,
                  origin="upper",interpolation="nearest")

        # Unknown cells (light grey tint)
        unk=np.zeros((*display.shape,4))
        unk[known==-1]=[0.7,0.7,0.8,0.2]
        ax.imshow(unk,origin="upper",interpolation="nearest",zorder=1)

        # Expanded cells
        if h["expanded"]:
            ov=np.zeros((*display.shape,4))
            r,g,b=int(exp_color[1:3],16)/255,int(exp_color[3:5],16)/255,int(exp_color[5:7],16)/255
            for er,ec in h["expanded"]:
                ov[er,ec]=[r,g,b,0.45]
            ax.imshow(ov,origin="upper",interpolation="nearest",zorder=2)

        # Path
        if h["path"] and len(h["path"])>1:
            pr=[p[0] for p in h["path"]]
            pc=[p[1] for p in h["path"]]
            ax.plot(pc,pr,"-",color=path_color,linewidth=2.5,zorder=4,alpha=0.9)
            ax.plot(pc,pr,".",color=path_color,markersize=3,zorder=4)

        # Trajectory
        traj=h["trajectory"]
        if len(traj)>1:
            ax.plot([p[1] for p in traj],[p[0] for p in traj],
                    "-",color="#ffffff",linewidth=1.0,alpha=0.4,zorder=3)

        # Agent & Target markers
        ar,ac=h["agent"]
        ax.plot(ac,ar,"o",color="#00e5b0",markersize=9,
                markeredgecolor="white",markeredgewidth=1.5,zorder=6)
        ax.text(ac,ar,"A",color="white",fontsize=5,
                ha="center",va="center",fontweight="bold",zorder=7)
        ax.plot(tc,tr,"o",color="#ff5370",markersize=9,
                markeredgecolor="white",markeredgewidth=1.5,zorder=6)
        ax.text(tc,tr,"T",color="white",fontsize=5,
                ha="center",va="center",fontweight="bold",zorder=7)

        # Title
        stype=h["type"]
        snum=h.get("search_num","")
        n_exp=len(h["expanded"])
        label={
            "search":      f"Search #{snum} | {n_exp} cells expanded",
            "move":        "Agent moving",
            "search_failed":"❌ No path found",
        }.get(stype,stype)
        ax.set_title(f"{title}\n{label}",
                     color=path_color,fontsize=9,pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    def _draw(self):
        w=self.world; tr,tc=w["target"]

        # True grid
        self.ax_true.cla(); self.ax_true.set_facecolor("#0d0d18")
        self.ax_true.imshow(w["grid"],cmap=CMAP_GRID,vmin=0,vmax=1,
                            origin="upper",interpolation="nearest")
        ar0,ac0=w["agent"]
        self.ax_true.plot(ac0,ar0,"o",color="#00e5b0",markersize=9,
                          markeredgecolor="white",markeredgewidth=1.5,zorder=5)
        self.ax_true.text(ac0,ar0,"A",color="white",fontsize=5,
                          ha="center",va="center",fontweight="bold",zorder=6)
        self.ax_true.plot(tc,tr,"o",color="#ff5370",markersize=9,
                          markeredgecolor="white",markeredgewidth=1.5,zorder=5)
        self.ax_true.text(tc,tr,"T",color="white",fontsize=5,
                          ha="center",va="center",fontweight="bold",zorder=6)
        self.ax_true.set_title("TRUE GRIDWORLD",color="#7c6af7",fontsize=10,pad=6)

        # Forward panel
        self._draw_panel(self.ax_fwd,self.fwd_hist,self.step,
                         "FORWARD A*  (agent → target)",
                         "#00e5b0","#00e5b0")

        # Backward panel
        self._draw_panel(self.ax_bwd,self.bwd_hist,self.step,
                         "BACKWARD A*  (target → agent)",
                         "#7c6af7","#7c6af7")

        # Legend
        patches=[
            mpatches.Patch(color="#00e5b0",alpha=0.5,label="Forward expanded"),
            mpatches.Patch(color="#7c6af7",alpha=0.5,label="Backward expanded"),
            mpatches.Patch(color="#00e5b0",label="Forward path"),
            mpatches.Patch(color="#7c6af7",label="Backward path"),
            mpatches.Patch(color="#ffffff",alpha=0.4,label="Trajectory"),
        ]
        self.ax_bwd.legend(handles=patches,loc="lower left",fontsize=6,
                           facecolor="#1a1a2e",edgecolor="#333",
                           labelcolor="white",framealpha=0.9)

        # Footer stats
        fs=self.results["forward"]["solver"]
        bs=self.results["backward"]["solver"]
        footer=(f"Step {self.step+1}/{max(len(self.fwd_hist),len(self.bwd_hist))}  |  "
                f"Forward:  {fs.total_expanded:,} expanded, "
                f"{fs.total_moves} moves, {fs.runtime*1000:.1f}ms  ‖  "
                f"Backward: {bs.total_expanded:,} expanded, "
                f"{bs.total_moves} moves, {bs.runtime*1000:.1f}ms")
        self.fig.texts.clear()
        self.fig.text(0.5,0.005,footer,ha="center",color="#6668a0",fontsize=8)
        self.fig.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    worlds=load_or_generate_worlds()

    print("\n"+"="*50)
    print("  PART 4 — FORWARD A* vs BACKWARD A*")
    print("="*50)
    print("\n  Options:")
    print("  1. Visualize step-by-step on one world")
    print("  2. Run comparison on ALL 30 worlds + chart")
    print("  3. Both")

    choice=input("\n  Enter choice [1/2/3]: ").strip()

    if choice in ("1","3"):
        idx_in=input("  World number (1-30) or Enter for world 1: ").strip()
        idx=(int(idx_in)-1) if idx_in.isdigit() else 0
        idx=max(0,min(NUM_WORLDS-1,idx))
        world=worlds[idx]
        results=run_both(world)
        fs=results["forward"]["solver"]
        bs=results["backward"]["solver"]
        print(f"\n  ┌──────────────┬────────────┬────────────┐")
        print(f"  │              │  Forward   │  Backward  │")
        print(f"  ├──────────────┼────────────┼────────────┤")
        print(f"  │ Expanded     │ {fs.total_expanded:>10,} │ {bs.total_expanded:>10,} │")
        print(f"  │ Moves        │ {fs.total_moves:>10,} │ {bs.total_moves:>10,} │")
        print(f"  │ Searches     │ {fs.num_searches:>10,} │ {bs.num_searches:>10,} │")
        print(f"  │ Runtime (ms) │ {fs.runtime*1000:>10.2f} │ {bs.runtime*1000:>10.2f} │")
        print(f"  └──────────────┴────────────┴────────────┘")
        print(f"\n  Launching visualizer...")
        FwdBwdVisualizer(world,results)

    if choice in ("2","3"):
        run_comparison_all(worlds)

    print("\n  Done!")
