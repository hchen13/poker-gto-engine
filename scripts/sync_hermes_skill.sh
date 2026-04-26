#!/bin/bash
# Sync the body of skill/poker/SKILL.md → ~/.hermes/skills/gaming/poker-gto/SKILL.md,
# preserving hermes's own frontmatter (name/tags/metadata).
#
# The project+claude-code versions are already hardlinked at
# /Users/claire/.claude/skills/poker/SKILL.md, so editing the project copy
# auto-updates claude-code. Only hermes needs a sync because its frontmatter
# schema differs (name: poker-gto, tags, metadata.hermes).

set -e
cd "$(dirname "$0")/.."

SOURCE=skill/poker/SKILL.md
HERMES=~/.hermes/skills/gaming/poker-gto/SKILL.md

if [ ! -f "$SOURCE" ] || [ ! -f "$HERMES" ]; then
    echo "Missing file. source=$SOURCE hermes=$HERMES" >&2
    exit 1
fi

# Extract hermes frontmatter (between first pair of '---' lines)
tmp=$(mktemp)
awk '
    /^---$/ { n++; print; next }
    n == 1 { print; next }
    n >= 2 { exit }
' "$HERMES" > "$tmp"

# Extract source body (after its frontmatter)
awk '
    /^---$/ { n++; next }
    n >= 2 { print }
' "$SOURCE" >> "$tmp"

# Only overwrite if content actually differs
if ! diff -q "$tmp" "$HERMES" > /dev/null; then
    mv "$tmp" "$HERMES"
    echo "Synced hermes SKILL.md body from project."
else
    rm "$tmp"
    echo "Hermes SKILL.md already in sync."
fi
