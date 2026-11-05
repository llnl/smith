# AI-Assisted Development Policy

## Purpose and scope

We encourage thoughtful use of AI tools, including agentic coding, to explore ideas, improve productivity, and strengthen software quality. Their use is optional. Contributions are evaluated on their quality and value, regardless of the tools used. Questions, mentoring, and early design discussions are always welcome.

This policy covers code, tests, documentation, build configuration, issue reports, and discussion responses. Expectations depend on whether a submission seeks exploratory feedback or implementation review.

## Avoid extractive contributions

An **extractive contribution**[^eghbal] costs the project more effort to review, correct, or maintain than the value it adds.

## Prepare contributions for review

Before requesting implementation review or proposing a pull request (PR) for acceptance, authors must personally review the complete diff and understand the technical details well enough to discuss and revise them.

If the PR is meant for exploration and discussion, clearly label it in the PR description to set expectations of the reviewers. A complete understanding is not required for this type of PR. State what is uncertain and what feedback you want.

## Keep pull requests reviewable

- Organize each PR around a single manageable concept
- Keep size reasonable
- Provide a good PR description and code comments
- Separate large unrelated fixes, formatting, and refactoring into their own PR

If splitting is impractical, explain why.

## Issue reports and responses

Personally review AI-generated issue reports. Identify unverified explanations as hypotheses. Never present AI-generated logs, test results, or reproduction claims as actual observations.

AI may help draft replies, but a human must review, understand, and post each response. **Do not configure agents to respond autonomously on PRs or issues.**

## Enforcement

Maintainers may require revisions or close impractical submissions. Repeated violations may result in a ban. Before escalating to a ban, maintainers will normally give contributors a reasonable opportunity to improve.

[^eghbal]: Adapted from Nadia Eghbal's discussion of extractive contributions in *Working in Public: The Making and Maintenance of Open Source Software* (Stripe Press, 2020).
