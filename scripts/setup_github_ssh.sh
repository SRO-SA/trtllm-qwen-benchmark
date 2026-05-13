#!/bin/bash
set -euo pipefail

KEY_PATH="$HOME/.ssh/github_trtllm_deploy_key"
REPO_SSH_URL="${REPO_SSH_URL:-git@github.com:SRO-SA/trtllm-qwen-benchmark.git}"
GIT_USER_NAME="${GIT_USER_NAME:-Soroosh}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

echo "Setting up GitHub deploy key..."

if [[ -n "${GITHUB_DEPLOY_KEY_B64:-}" ]]; then
    echo "$GITHUB_DEPLOY_KEY_B64" | base64 -d > "$KEY_PATH"
else
    echo "Paste your BASE64-encoded private deploy key below."
    echo "It will not be shown on screen."
    read -rsp "GITHUB_DEPLOY_KEY_B64: " KEY_B64
    echo
    echo "$KEY_B64" | base64 -d > "$KEY_PATH"
fi

chmod 600 "$KEY_PATH"

cat > "$HOME/.ssh/config" <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile $KEY_PATH
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF

chmod 600 "$HOME/.ssh/config"

# Add GitHub host key if not already present
ssh-keyscan github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 644 "$HOME/.ssh/known_hosts"

echo "Testing GitHub SSH connection..."
ssh -T git@github.com || true

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Updating git remote to SSH:"
    git remote set-url origin "$REPO_SSH_URL"
    git remote -v
fi

git config --global user.name "$GIT_USER_NAME"

if [[ -n "$GIT_USER_EMAIL" ]]; then
    git config --global user.email "$GIT_USER_EMAIL"
fi

echo "GitHub SSH setup finished."
