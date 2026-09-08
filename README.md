# NumericalPDE — Team 11

MTH321 Project 1: numerical methods for Ordinary Differential Equation (ODE)
initial value problems.

## Repository layout

| Path        | Purpose                                          |
| ----------- | ------------------------------------------------ |
| `code/`     | All source code (`integrators.py`, `run_all.py`, …) |
| `figures/`  | Generated figures (final ones go in `report/`)   |
| `report/`   | LaTeX report + slides                            |
| `notes/`    | Meeting notes, issue tickets                     |

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

| Member        | GitHub      | Role                    |
| ------------- | ----------- | ----------------------- |
| [Name]        | [username]  | Project Manager         |
| [Name]        | [username]  | Integrators / methods   |
| [Name]        | [username]  | Plotting / figures      |
| [Name]        | [username]  | Report / slides         |

## Git workflow

```bash
git pull            # get the latest at the start of a session
# … edit files …
git add -A
git commit -m "Describe what changed"
git push
```

Commit at the end of every session with a message that says what changed.
