# Copilot Code Review Instructions

When reviewing a pull request, determine whether the changes are user-facing.

User-facing changes include changes that affect behavior, output, interfaces, configuration, command-line options, APIs, workflows, defaults, error messages, or other functionality visible to users.

If a pull request contains a user-facing change, verify that `RELEASE_NOTES.md` includes an appropriate entry describing the change.

The release note should:
- Be very short: normally 1–2 sentences.
- Describe the change from the user's perspective.
- Focus on what changed and why it matters to users.
- Avoid implementation details unless they are necessary for understanding the user-visible behavior.

If the pull request contains a user-facing change but does not include an appropriate `RELEASE_NOTES.md` entry, leave a review comment requesting one.

Do not request a release note for changes that are purely internal, such as refactoring, tests, CI changes, formatting, comments, or implementation changes that do not alter user-visible behavior.
