#!/usr/bin/env bash
#
# Neme-Anima one-click installer
# ------------------------------
#
# Walks an entry-level user from a fresh checkout to a running UI:
#   1. installs uv (Python package manager) if missing
#   2. installs Node.js + npm if missing (apt with sudo, or nvm fallback)
#   3. installs Python deps with the GPU extras (`uv sync --group gpu`)
#   4. builds the frontend bundle
#   5. clones tdrussell/diffusion-pipe into $DIFFUSION_PIPE_DIR (default: ~/diffusion-pipe)
#   6. creates a venv inside that clone and installs its requirements
#   7. downloads the three Anima training weight files from HuggingFace
#   8. writes ~/.neme-anima/training-defaults.json so the UI's Settings tab
#      shows the four trainer paths pre-filled for new projects
#   9. launches `uv run neme-anima ui`
#
# Usage: bash install_and_run.sh
#
# The script is idempotent: re-running skips work that's already done.
#
# Environment variables you can override:
#   DIFFUSION_PIPE_DIR  where to clone diffusion-pipe (default: ~/diffusion-pipe)
#   MODELS_DIR          where to store downloaded weights (default: ~/.cache/neme-anima/models)
#   SKIP_MODELS=1       skip the ~14 GB weight download (useful for testing)
#   SKIP_LAUNCH=1       install everything but don't start the UI at the end
#

set -euo pipefail

# ----- terminal styling -----------------------------------------------------

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; RESET=""
fi

step()    { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
info()    { printf '    %s\n' "$*"; }
success() { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()    { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
fail()    { printf '\n%sError:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# ----- preconditions --------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]] || ! grep -q '^name = "neme-anima"' pyproject.toml; then
    fail "this script must live at the root of the neme-extractor repo"
fi

DIFFUSION_PIPE_DIR="${DIFFUSION_PIPE_DIR:-$HOME/diffusion-pipe}"
MODELS_DIR="${MODELS_DIR:-$HOME/.cache/neme-anima/models}"
DEFAULTS_DIR="$HOME/.neme-anima"
DEFAULTS_FILE="$DEFAULTS_DIR/training-defaults.json"

DIT_FILE="$MODELS_DIR/anima-preview3-base.safetensors"
VAE_FILE="$MODELS_DIR/qwen_image_vae.safetensors"
LLM_FILE="$MODELS_DIR/qwen_3_06b_base.safetensors"

printf '%s%sNeme-Anima installer%s\n' "$BOLD" "$CYAN" "$RESET"
info "repo:           $REPO_ROOT"
info "diffusion-pipe: $DIFFUSION_PIPE_DIR"
info "models:         $MODELS_DIR"

# ----- 1. uv ----------------------------------------------------------------

step "1/9  Checking for uv"

if ! command -v uv >/dev/null 2>&1; then
    info "installing uv via the official script…"
    if ! command -v curl >/dev/null 2>&1; then
        fail "curl is required to install uv. Install curl first (e.g. 'sudo apt install curl')."
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer writes to ~/.local/bin or ~/.cargo/bin; pull both onto PATH
    # for the rest of this script even if the user hasn't restarted their shell.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    fail "uv install failed. Try installing manually: https://docs.astral.sh/uv/getting-started/installation/"
fi
success "uv $(uv --version | awk '{print $2}')"

# ----- 2. node + npm --------------------------------------------------------

step "2/9  Checking for Node.js + npm"

need_node=true
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    node_major="$(node -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
    if [[ -n "$node_major" && "$node_major" -ge 18 ]]; then
        need_node=false
    else
        warn "found node $(node -v) — too old; need >= v18"
    fi
fi

if $need_node; then
    if [[ "$(uname -s)" != "Linux" ]]; then
        fail "Node.js auto-install only supports Linux. Install Node 20+ from https://nodejs.org and re-run."
    fi
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && command -v apt-get >/dev/null 2>&1; then
        info "installing Node.js 20.x via NodeSource (apt)…"
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        info "no passwordless sudo available — falling back to nvm (per-user install)"
        export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
        if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
            curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        fi
        # shellcheck disable=SC1091
        . "$NVM_DIR/nvm.sh"
        nvm install --lts
        nvm use --lts
    fi
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    fail "Node.js install failed. Install Node 20+ manually and re-run."
fi
success "node $(node -v) / npm $(npm -v)"

# ----- 3. python deps -------------------------------------------------------

step "3/9  Installing Python dependencies (uv sync --group gpu)"
info "this fetches PyTorch CUDA wheels — first run may take several minutes…"
uv sync --group gpu
success "Python deps installed"

# ----- 4. frontend ----------------------------------------------------------

step "4/9  Building the frontend"
pushd frontend >/dev/null
if [[ ! -d node_modules ]] || [[ package.json -nt node_modules ]]; then
    info "running npm install…"
    npm install
else
    info "node_modules up to date"
fi
info "running npm run build…"
npm run build
popd >/dev/null
success "frontend bundle built into src/neme_anima/server/static"

# ----- 5. clone diffusion-pipe ---------------------------------------------

step "5/9  Cloning tdrussell/diffusion-pipe"
if [[ -d "$DIFFUSION_PIPE_DIR/.git" ]]; then
    info "already present at $DIFFUSION_PIPE_DIR — pulling latest"
    git -C "$DIFFUSION_PIPE_DIR" pull --ff-only || warn "git pull failed (continuing with current checkout)"
else
    git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe "$DIFFUSION_PIPE_DIR"
fi
if [[ ! -f "$DIFFUSION_PIPE_DIR/train.py" ]]; then
    fail "diffusion-pipe clone is missing train.py at $DIFFUSION_PIPE_DIR"
fi
success "diffusion-pipe at $DIFFUSION_PIPE_DIR"

# ----- 6. diffusion-pipe venv ----------------------------------------------

step "6/9  Setting up diffusion-pipe's Python venv"
pushd "$DIFFUSION_PIPE_DIR" >/dev/null
if [[ ! -d .venv ]]; then
    info "creating venv with uv…"
    uv venv
fi
info "installing diffusion-pipe requirements (this can take a while)…"
if [[ -f requirements.txt ]]; then
    uv pip install -r requirements.txt
else
    warn "no requirements.txt found in $DIFFUSION_PIPE_DIR — skipping (upstream layout may have changed)"
fi
# torchvision is required by diffusion-pipe but omitted from some upstream
# requirements.txt versions; install it explicitly to avoid import errors at
# training time.
if ! .venv/bin/python -c "import torchvision" 2>/dev/null; then
    info "torchvision missing from diffusion-pipe venv — installing…"
    uv pip install torchvision
fi
# deepspeed is the default launcher; verify it landed.
if [[ ! -x .venv/bin/deepspeed ]] && [[ ! -x venv/bin/deepspeed ]]; then
    warn "deepspeed binary not found in the venv — training may fail to launch."
    warn "you may need to: cd '$DIFFUSION_PIPE_DIR' && uv pip install deepspeed"
fi
popd >/dev/null
success "diffusion-pipe venv ready"

# ----- 7. download model weights --------------------------------------------

step "7/9  Downloading Anima training weights"
if [[ "${SKIP_MODELS:-}" == "1" ]]; then
    warn "SKIP_MODELS=1 set — skipping ~14 GB download"
else
    mkdir -p "$MODELS_DIR"
    # Use the `hf` CLI shipped by huggingface-hub (>=0.29). Older releases
    # only have `huggingface-cli`; fall back to that.
    HF_BIN=""
    if uv run --quiet -- hf --help >/dev/null 2>&1; then
        HF_BIN="hf"
    elif uv run --quiet -- huggingface-cli --help >/dev/null 2>&1; then
        HF_BIN="huggingface-cli"
    else
        fail "neither 'hf' nor 'huggingface-cli' is available. Re-run 'uv sync --group gpu'."
    fi

    download_one() {
        local repo_path="$1" dest="$2" label="$3"
        if [[ -f "$dest" && -s "$dest" ]]; then
            success "$label already present ($(du -h "$dest" | awk '{print $1}'))"
            return
        fi
        info "downloading $label …"
        # Strategy: download the single file from the repo into a temp staging
        # directory, then move it to MODELS_DIR with the flat name the docs
        # use. We avoid `--local-dir` directly on MODELS_DIR because the repo
        # nests files under split_files/<kind>/<file>.
        local tmpdir
        tmpdir="$(mktemp -d)"
        # `hf download` (and `huggingface-cli download`) accept positional args:
        # <repo_id> <filename> --local-dir <dir>.
        if ! uv run --quiet -- "$HF_BIN" download circlestone-labs/Anima \
                "$repo_path" --local-dir "$tmpdir" >/dev/null; then
            rm -rf "$tmpdir"
            fail "failed to download $repo_path from circlestone-labs/Anima.
   If you hit a 401/403, run 'uv run -- hf auth login' and retry.
   If your network is the bottleneck, set SKIP_MODELS=1 and grab the files
   manually into $MODELS_DIR (filenames: anima-preview3-base.safetensors,
   qwen_image_vae.safetensors, qwen_3_06b_base.safetensors)."
        fi
        local src="$tmpdir/$repo_path"
        if [[ ! -f "$src" ]]; then
            # Some hf-cli versions place the file at the leaf basename.
            src="$tmpdir/$(basename "$repo_path")"
        fi
        if [[ ! -f "$src" ]]; then
            rm -rf "$tmpdir"
            fail "downloader didn't produce the expected file for $repo_path"
        fi
        mv "$src" "$dest"
        rm -rf "$tmpdir"
        success "$label saved to $dest"
    }

    download_one "split_files/diffusion_models/anima-preview3-base.safetensors" \
                 "$DIT_FILE" "Anima DiT"
    download_one "split_files/vae/qwen_image_vae.safetensors" \
                 "$VAE_FILE" "Qwen image VAE"
    download_one "split_files/text_encoders/qwen_3_06b_base.safetensors" \
                 "$LLM_FILE" "Qwen 3 0.6B text encoder"
fi

# ----- 8. write training-defaults.json --------------------------------------

step "8/9  Writing training defaults to $DEFAULTS_FILE"
mkdir -p "$DEFAULTS_DIR"
cat > "$DEFAULTS_FILE" <<JSON
{
  "diffusion_pipe_dir": "$DIFFUSION_PIPE_DIR",
  "dit_path":           "$DIT_FILE",
  "vae_path":           "$VAE_FILE",
  "llm_path":           "$LLM_FILE"
}
JSON
success "defaults written — new projects will load these in the Settings tab"

# ----- 9. launch ------------------------------------------------------------

step "9/9  Launching the UI"
if [[ "${SKIP_LAUNCH:-}" == "1" ]]; then
    info "SKIP_LAUNCH=1 set — install complete, not starting the UI."
    info "to start it later, run: ${BOLD}uv run neme-anima ui${RESET}"
    exit 0
fi
info "starting ${BOLD}uv run neme-anima ui${RESET} (Ctrl+C to stop)…"
exec uv run neme-anima ui
