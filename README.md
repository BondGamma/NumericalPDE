# NumericalODE — Team 11
```text
╔══════════════╗     y
║ Solving SDE  ║     │        ╱╲
║    💻 ⚙️     ║     │      ╱    ╲
╚══════════════╝     │   ╱╲╱      ╲
                     │ ╱            ╲
                     └───────────────► t
```


MTH321 Project 1: numerical methods for Ordinary Differential Equation (ODE)
initial value problems.

## Repository layout

```
project-repo/
├── README.md          # how to run the code + who does what
├── .gitignore
├── code/              # all source
│   ├── run_all.py     # entry point that runs every experiment
│   ├── SDEs/          # model definitions + Brownian-motion engine
│   │   ├── BM_engine.py   # standard_bm, correlated_bm_pair, extract_dw
│   │   ├── gbm.py         # geometric Brownian motion (SDE)
│   │   └── nasv.py        # noise-assisted stochastic volatility (SDE)
│   ├── solvers/       # time-stepping schemes
│   │   ├── em.py          # Euler–Maruyama
│   │   └── milstein.py    # Milstein
│   ├── tools/         # shared helpers
│   │   └── mc.py          # Monte-Carlo BM simulator (writes logs to data/)
│   └── experiments/   # one script per experiment
│       └── exp0_simulate_bm.py
├── data/              # generated BM increment logs (gitignored)
├── figures/           # generated figures (final ones go in the report)
├── report/            # LaTeX report + slides
└── notes/             # meeting notes, issue tickets (optional)
```

Keep `code/` for source only — never commit output files (`data/` is ignored,
see `.gitignore`).

## How to run the code

```bash
# 1. clone
git clone <repo-url>
cd NumericalPDE

# 2. create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

# 3. install dependencies
pip install numpy scipy matplotlib

# 4. run everything
python code/run_all.py
```

Generated figures are written to `figures/`; the final ones get copied into
`report/`. LaTeX build artefacts (`.aux`, `.log`, `.toc`, `.bbl`, `.blg`) are
ignored by `.gitignore`.

## Experiments

### Experiment 0 — "same Brownian motion" smoke test

Verifies the core assumption behind the whole convergence study: that every
step size `dt` sees the **same underlying Brownian path**. If that fails, a
strong-convergence comparison across `dt` is meaningless, so this is checked
first.

The pipeline it exercises:

1. `tools/mc.simulate` draws fine Brownian increments **once** at the maximum
   resolution `n_max` and saves them to `data/`:
   - `"standard"` → `data/gbm_dw_log.npy` (independent standard BM, for GBM)
   - `"pairing"`  → `data/nasv_dw_log.npy` (correlated pair, for NA-SV)
2. `SDEs/BM_engine.extract_dw` coarsens a log to any smaller step count `n`
   (which must divide `n_max`) by summing blocks of fine increments, so the
   coarse path is exactly the fine path sampled on the coarser grid.
3. The script plots each fine path and overlays the coarse samples at the
   chosen `n`'s — they must coincide, confirming the unification.

Run it:

```bash
python code/experiments/exp0_simulate_bm.py
```

Output: 4 PNGs in `figures/` — 2 simulations × 2 engines
(`bm_standard_sim{1,2}.png`, `bm_pairing_sim{1,2}.png`).

It also exercises the advanced path plotters (`code/visualizations/paths.py`)
on every 1-D model — pure Brownian motion, exact GBM, EM/Milstein GBM, and
NA-SV — emitting a confidence-band figure (`band_<model>.png`) and a
cross-section figure (`slices_<model>.png`) for each.

## Who does what

| Member        | Username      | Role                      |
|---------------|---------------|---------------------------|
| Zizhao Wang   | Pwzza         | Project manager           |
| [Member 2]    | [username 2]  | Mathematical theory       |
| Artem Bobrov  | BondGamma     | Algorithm implementation  |
| [Member 4]    | [username 4]  | Visualization & report    |
| [Member 5]    | [username 5]  | Testing & validation      |

What each role owns:

- **Project manager** — plan, meetings, presentations, weekly records
- **Mathematical theory** — model, stability regions, Jacobian, error estimates
- **Algorithm implementation** — G / RK4 / implicit / Newton / adaptivity + git
- **Visualization & report** — figures, animations, LaTeX report, slides
- **Testing & validation** — building your own oracle, cross-checks, edge cases
