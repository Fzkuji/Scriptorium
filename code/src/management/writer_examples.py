"""Worked shell examples appended to the writer task for weaker models.

The writing protocol states that each shell call is one independent
transaction. Stronger models infer the consequence; gpt-4o-mini does not, and
falls into a stall: it appends the prose in one call and the footnote
definition in the next, so the first call is rejected for an undefined
footnote, the second for a fact whose paragraph no longer exists, and it
retries the same pair until the turn budget runs out.

These examples make the consequence concrete — write the whole file in a single
call, with prose and footnote definitions together — and show the failure mode
explicitly so the model can recognise it.
"""

WRITER_SHELL_EXAMPLES = """
Worked examples of shell usage. Follow this shape.

Each shell call is validated and committed on its own. A paragraph and the
footnote definitions it cites must therefore appear in the SAME call. Writing
prose in one call and its definition in the next fails both times.

Correct — one call writes the complete file:

cat > topics/hobbies.md <<'EOF'
# Hobbies

## Painting

Melanie painted a lakeside landscape and found it calming.[^new-evidence-painting] ^new-block-painting

[^new-evidence-painting]: Time: `2023-05-08`; Sources: leaderboard/session-7/msg-4
EOF

Correct — extending an existing file, rewritten whole in one call:

cat > topics/hobbies.md <<'EOF'
# Hobbies

## Painting

Melanie painted a lakeside landscape and found it calming.[^e-1a2b3c4d5e] ^a1b2c3d4

Melanie sold the painting to a neighbour.[^new-evidence-sale] ^new-block-sale

[^e-1a2b3c4d5e]: Time: `2023-05-08`; Sources: leaderboard/session-7/msg-4
[^new-evidence-sale]: Time: `2023-06-02`; Sources: leaderboard/session-9/msg-2
EOF

Wrong — the fact cites a footnote this call does not define:

echo 'Melanie sold the painting.[^new-evidence-sale] ^new-block-sale' >> topics/hobbies.md

Wrong — the definition arrives in a later call, after the first was rejected:

echo '[^new-evidence-sale]: Time: `2023-06-02`; Sources: leaderboard/session-9/msg-2' >> topics/hobbies.md

If a call is rejected, do not repeat it. Read the error, then rewrite the whole
file correctly in one call.

Read before writing so existing content is preserved:

cat topics/hobbies.md

Source handles are the bracketed identifiers on each line of the conversation
below. Copy a handle exactly as it appears there, character for character. Do
not prepend a word to it, do not reformat it, and never invent one. The handles
shown in these examples are illustrations; use the real ones from the input.
"""
