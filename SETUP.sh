#!/usr/bin/env bash
# volbench SETUP for Linux (the 4090 box). Idempotent — safe to re-run.
#
#   bash SETUP.sh                          # gh is authenticated
#   bash SETUP.sh --user <github-user>     # gh not authenticated
#   bash SETUP.sh --scaffold /path/to/06_volbench_scaffold [--user <u>]
#                                          # seed an EMPTY GitHub repo from scaffold files
#   VOLBENCH_ROOT=/data/martin bash SETUP.sh   # override where the repo lives
#
# Creates:
#   $VOLBENCH_ROOT/volbench                      main checkout      -> T0 and D
#   $VOLBENCH_ROOT/volbench-wt/{data,models,eval}  stream worktrees -> terminals A / B / C

set -euo pipefail

LOGIN=""; SCAFFOLD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --user)     LOGIN="${2:-}"; shift 2 ;;
        --scaffold) SCAFFOLD="${2:-}"; shift 2 ;;
        *)          LOGIN="$1"; shift ;;      # bare arg = username (back-compat)
    esac
done

say()  { printf '\033[36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------- 0. sane install root
# Guard against a surprising $HOME (containers, sudo, service accounts): a bare
# /home or / would scatter the repo across a shared directory.
if [ -n "${VOLBENCH_ROOT:-}" ]; then
    ROOT="${VOLBENCH_ROOT}"
elif [ -n "${HOME:-}" ] && [ "${HOME}" != "/home" ] && [ "${HOME}" != "/" ]; then
    ROOT="${HOME}"
else
    ROOT="/home/$(id -un)"
    warn "\$HOME is '${HOME:-unset}', which looks wrong — using ${ROOT} instead."
    warn "Override with: VOLBENCH_ROOT=/your/path bash SETUP.sh"
fi
mkdir -p "${ROOT}"
REPO="${ROOT}/volbench"
WT="${ROOT}/volbench-wt"
say "Install root: ${ROOT}"

# ------------------------------------------------------------- 1. toolchain
command -v git >/dev/null || die "git missing: sudo apt install -y git"

if ! command -v uv >/dev/null; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${ROOT}/.local/bin:${HOME:-$ROOT}/.local/bin:${PATH}"
command -v uv >/dev/null || die "uv installed but not on PATH — open a new shell and re-run."

if [ -z "${LOGIN}" ] && command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    LOGIN="$(gh api user --jq .login 2>/dev/null || true)"
fi
[ -n "${LOGIN}" ] || die "Could not determine your GitHub username.
  Run:  bash SETUP.sh --user <your-github-username>
  (replace <your-github-username> with your ACTUAL login, e.g. martin-nlp)"
case "${LOGIN}" in
    GITHUB|github|USER|user|CHANGEME|"<"*)
        die "'${LOGIN}' is a placeholder, not a GitHub username. Re-run with --user <your-actual-login>." ;;
esac
say "GitHub user: ${LOGIN}"

# ------------------------------------------------------------------ 2. host
if command -v nvidia-smi >/dev/null; then
    say "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
else
    warn "nvidia-smi not found — fine for Phases 1–2 (CPU only)."
fi
say "CPU: $(nproc) logical cores | RAM: $(free -g | awk '/^Mem:/{print $2}') GB"

# ----------------------------------------------------------------- 3. clone
if [ -d "${REPO}/.git" ]; then
    say "Repo exists at ${REPO} — fetching."
    git -C "${REPO}" fetch --all --prune
else
    say "Cloning ${LOGIN}/volbench"
    git clone "https://github.com/${LOGIN}/volbench.git" "${REPO}" 2>&1 || \
        die "Clone failed. Does https://github.com/${LOGIN}/volbench exist?"
fi
cd "${REPO}"

# ------------------------------------------- 4. empty repo? seed it or explain
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    warn "The GitHub repo has no commits yet."
    if [ -z "${SCAFFOLD}" ]; then
        die "Nothing to build from. Choose one:
  (a) On the Windows box run 06_volbench_scaffold\\SETUP.ps1 (it seeds and pushes), then re-run this script; or
  (b) Copy the 06_volbench_scaffold folder to this machine and re-run:
        bash SETUP.sh --user ${LOGIN} --scaffold /path/to/06_volbench_scaffold"
    fi
    [ -d "${SCAFFOLD}/src/volbench" ] || die "--scaffold '${SCAFFOLD}' does not look like 06_volbench_scaffold (no src/volbench)."

    say "Seeding from ${SCAFFOLD}"
    git checkout -b main 2>/dev/null || git checkout main
    ( cd "${SCAFFOLD}" && tar --exclude=SETUP.ps1 --exclude=SETUP.sh --exclude=bootstrap.ps1 \
        --exclude=_transfer --exclude=SCAFFOLD_README.md -cf - . ) | tar -xf - -C "${REPO}"

    # files the Cowork bridge cannot write under their real names
    mkdir -p "${REPO}/.github/workflows"
    [ -f "${REPO}/Makefile" ]                || cp "${SCAFFOLD}/_transfer/Makefile.txt"               "${REPO}/Makefile"
    [ -f "${REPO}/.pre-commit-config.yaml" ] || cp "${SCAFFOLD}/_transfer/pre-commit-config.yaml.txt" "${REPO}/.pre-commit-config.yaml"
    [ -f "${REPO}/.github/workflows/ci.yml" ]|| cp "${SCAFFOLD}/_transfer/ci.yml.txt"                 "${REPO}/.github/workflows/ci.yml"

    # skills live in .claude/skills (dotless in the scaffold for transfer)
    if [ -d "${REPO}/claude-skills" ] && [ ! -d "${REPO}/.claude/skills" ]; then
        mkdir -p "${REPO}/.claude"
        mv "${REPO}/claude-skills" "${REPO}/.claude/skills"
    fi
    rm -rf "${REPO}/claude-skills"

    sed -i "s/CHANGEME/${LOGIN}/g" "${REPO}/pyproject.toml"
    say "Seeded."
else
    git checkout main 2>/dev/null || git checkout -b main
    git pull --ff-only origin main 2>/dev/null || true
fi

# --------------------------------------------------------- 5. install/check
say "Installing dependencies"
uv sync --dev
say "Running checks"
uv run pytest -q
uv run ruff check .
uv run mypy

if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "chore: volbench scaffold - Distribution, RollingOriginSplitter, metrics, CI"
    git push -u origin main
    say "Committed and pushed."
fi

# -------------------------------------------------------------- 6. worktrees
mkdir -p "${WT}"
setup_wt() {
    local name="$1" branch="$2" path="${WT}/$1"
    if [ -d "${path}" ]; then say "Worktree '${name}' exists — skipping."; return; fi
    if git show-ref --verify --quiet "refs/heads/${branch}"; then :
    elif git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
        git branch --track "${branch}" "origin/${branch}"
    else git branch "${branch}" main; fi
    git worktree add "${path}" "${branch}"
    ( cd "${path}" && uv sync --dev && uv run pytest -q )
    git push -u origin "${branch}" 2>/dev/null || warn "Could not push ${branch} (push it later)."
    say "Worktree '${name}' ready on ${branch}."
}
setup_wt data   feat/data-layer
setup_wt models feat/model-adapters
setup_wt eval   feat/evaluation

# --------------------------------------------------------------- 7. summary
printf '\n'
say "SETUP COMPLETE"
cat <<EOF
  repo     : ${REPO}
  github   : https://github.com/${LOGIN}/volbench
  prompts  : ${REPO}/docs/phase1_prompts.md

  T0 / D   : cd ${REPO}        && claude
  stream A : cd ${WT}/data     && claude    # data layer
  stream B : cd ${WT}/models   && claude    # model adapters
  stream C : cd ${WT}/eval     && claude    # evaluation

  Then claim the package name:  cd ${REPO} && uv build && uv publish
EOF
