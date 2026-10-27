# AI-Assisted Development Policy

## Purpose and scope

We encourage thoughtful use of AI tools, including agentic coding, to explore ideas, improve productivity, and strengthen software quality. Their use is optional. Contributions are evaluated on their quality and value, regardless of the tools used.

This policy covers code, tests, documentation, build configuration, issue reports, and discussion responses. Expectations depend on whether a submission seeks exploratory feedback or implementation review.

## Prepare contributions for review

Before requesting implementation review or proposing a pull request (PR) for acceptance, authors must:

- **Understand the changes.** Explain their purpose, implementation, assumptions, and limitations, and understand the technical details well enough to discuss and revise them.
- **Personally review the complete diff.** Check code, tests, documentation, and generated files; correct mistakes and remove unnecessary changes. Review subsequent revisions before requesting another review. AI-assisted review supplements this work; it does not replace it.
- **Validate the work.** Run appropriate tests and checks, inspect the results, and report validation gaps. For scientific or numerical changes, address mathematical assumptions, physical behavior, and numerical accuracy. Passing tests or AI assurances alone do not establish correctness.
- **Verify supporting claims.** Check explanations in PR descriptions, comments, and documentation against the implementation and available evidence.

## Keep pull requests reviewable

PRs must allow reviewers to assess their purpose, implementation, and validation with reasonable effort.

- **Focus scope.** Organize each PR around a coherent problem or manageable concept. Separate unrelated fixes, formatting, and refactoring; remove redundant code and unnecessary generated material.
- **Keep size reasonable.** Consider conceptual complexity as well as line count. Split large work into smaller PRs with clear boundaries and dependencies. If splitting is impractical, explain why and provide a review plan.
- **Provide context.** Explain the problem, approach, key design choices, expected behavior, validation results, and limitations. Suggest a review order when helpful and identify areas needing specialist expertise.

## Exploration and discussion

Questions, mentoring, and early design discussions are welcome. A complete understanding or validated implementation is not required to open an exploratory issue, start a discussion, or share a draft PR for conceptual feedback.

Explicitly label the submission as exploratory, identify uncertainties, distinguish observations from hypotheses, and state what feedback you seek. Personally read the material before sharing it and keep it focused. Before requesting implementation review or acceptance, meet the requirements above. Discuss substantial or uncertain work with maintainers before investing in a full implementation.

## Issue reports and responses

Personally review AI-generated issue reports. Include supporting evidence, reproduction steps, and environment details when available; state whether you reproduced the behavior. Identify unverified explanations as hypotheses. Never present generated logs, test results, or reproduction claims as actual observations.

AI may help draft replies, but a human must review, understand, and post each response. Clearly identify uncertainties in exploratory replies. **Do not configure agents to respond autonomously on PRs or issues.**

## Avoid extractive contributions

An **extractive contribution** demands more project effort than the value it adds, particularly by shifting avoidable verification, correction, or maintenance work onto others. This applies with or without AI use.

Examples include unreviewed generated PRs, speculative issue reports presented as confirmed problems, repeated revisions that ignore feedback, and large volumes of low-value code or documentation. Unnecessary features, cleanups, or abstractions can also create disproportionate review and maintenance costs.

Clearly identified exploration and good-faith requests for help are not, by themselves, extractive. Contributors should explain what they have tried, what remains uncertain, and what help they need.

## Enforcement

Maintainers may request explanation, validation, simplification, or splitting before continuing review, and may defer or close extractive or impractical submissions.

Before escalating repeated violations to suspension or a ban, maintainers should ordinarily explain the problem, identify the expected correction, and give the contributor a reasonable opportunity to improve. Repeated extractive submissions or other repeated violations may result in suspension or a ban from project participation, at maintainer discretion.
