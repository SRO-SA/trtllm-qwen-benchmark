#!/bin/bash
set -euo pipefail

KEY_PATH="${KEY_PATH:-$HOME/.ssh/github_trtllm_deploy_key}"
REPO_SSH_URL="${REPO_SSH_URL:-git@github.com:SRO-SA/trtllm-qwen-benchmark.git}"
GIT_USER_NAME="${GIT_USER_NAME:-Soroosh}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

echo "Setting up GitHub deploy key..."

if [[ -n "${GITHUB_DEPLOY_KEY_FILE:-}" ]]; then
    echo "Using private key from file: $GITHUB_DEPLOY_KEY_FILE"
    cp "$GITHUB_DEPLOY_KEY_FILE" "$KEY_PATH"

elif [[ -n "${GITHUB_DEPLOY_KEY_B64:-}" ]]; then
    echo "Using base64-encoded private key from GITHUB_DEPLOY_KEY_B64"
    echo "$GITHUB_DEPLOY_KEY_B64" | tr -d ' \n\r\t' | base64 -d > "$KEY_PATH"

elif [[ -n "${GITHUB_DEPLOY_KEY:-}" ]]; then
    echo "Using raw private key from GITHUB_DEPLOY_KEY"
    printf "%s\n" "$GITHUB_DEPLOY_KEY" > "$KEY_PATH"

else
    echo "Paste your RAW PRIVATE deploy key below."
    echo "It should start with:"
    echo "  -----BEGIN OPENSSH PRIVATE KEY-----"
    echo
    echo "After pasting the full key, press Ctrl+D."
    echo
    cat > "$KEY_PATH"
fi

chmod 600 "$KEY_PATH"

if ! ssh-keygen -y -f "$KEY_PATH" >/dev/null 2>&1; then
    echo "ERROR: The saved key is not a valid private SSH key."
    echo "Common causes:"
    echo "  - You pasted the public key instead of the private key."
    echo "  - The key was partially copied."
    echo "  - Extra prompt/output text was pasted into the key."
    echo
    echo "Private key starts with: -----BEGIN OPENSSH PRIVATE KEY-----"
    echo "Public key starts with:  ssh-ed25519 ..."
    exit 1
fi

cat > "$HOME/.ssh/config" <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile $KEY_PATH
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF

chmod 600 "$HOME/.ssh/config"

ssh-keyscan github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 644 "$HOME/.ssh/known_hosts"

echo "Testing GitHub SSH connection..."
set +e
SSH_OUT="$(ssh -T git@github.com 2>&1)"
SSH_CODE=$?
set -e
echo "$SSH_OUT"

if echo "$SSH_OUT" | grep -qi "successfully authenticated"; then
    echo "GitHub SSH authentication succeeded."
elif echo "$SSH_OUT" | grep -qi "does not provide shell access"; then
    echo "GitHub SSH authentication appears successful."
else
    echo "WARNING: GitHub SSH authentication may have failed."
    echo "You can still inspect with: ssh -vT git@github.com"
fi

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
