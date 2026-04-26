# Installing the `/poker` skill

The repo ships with a one-shot installer at the project root:

```bash
./install.sh
```

This builds the Rust solver, installs the `poker` CLI on PATH (`~/.local/bin/poker`),
and runs a smoke test. See `AGENTS.md` for the LLM-operator-friendly version,
or `install.py --help` for installer flags.

## Per-agent skill registration

The installer does **not** modify your agent platform's skill directory.
Wire the skill in by hand:

### Claude Code

```bash
mkdir -p ~/.claude/skills
ln -sf "$(pwd)/skill/poker" ~/.claude/skills/poker
```

Then in Claude Code: `/poker <hand description>` or just describe a hand naturally.

### Hermes

```bash
cp -R skill/poker ~/.hermes/skills/gaming/poker-gto
# Or for a specific profile: ~/.hermes/profiles/<name>/skills/gaming/poker-gto
```

Hermes uses different SKILL.md frontmatter — keep its existing frontmatter
when refreshing the body from this repo.

### openclaw / other agents

The skill contract is the `poker` CLI. Any agent that can invoke shell
commands and read JSON can use it. Point the agent at `SKILL.md` for usage
instructions.

## What the skill does (high level)

1. Reads the user's natural-language hand description
2. Parses to structured spot (board, stacks, action history, hero cards, opponent type)
3. Calls `poker query` (precomputed lookup) or `poker solve-*` (on-demand)
4. Walks the solver output to the user's specific decision node + hero combo
5. Renders a `{action, 频率, 理由}` table; LLM writes the 理由 column

## Files in the skill

- `SKILL.md` — the skill definition (what the LLM reads)
- `cli.py` — `poker` CLI dispatcher
- `bin/*.py` — one .py per subcommand
- `lib/*.py` — shared helpers (path resolution, format, solver glue)
- `state/profiles/` — built-in opponent archetypes (fish/nit/reg/maniac)
- `state/` — per-player profile markdown files (user-managed)
