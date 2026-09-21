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
│   ├── integrators.py
│   └── run_all.py
├── figures/           # generated figures (final ones go in the report)
├── report/            # LaTeX report + slides
└── notes/             # meeting notes, issue tickets (optional)
```

Keep `code/` for source only — never commit output files (see `.gitignore`).

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
