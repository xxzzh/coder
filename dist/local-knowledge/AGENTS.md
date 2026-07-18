# Project Instructions

- Install Codex skills for this project into this repository's `.codex\skills` directory.
- Use the skill installer with `--dest <project-root>\.codex\skills`.
- Do not install project-specific skills into `%USERPROFILE%\.codex\skills` unless the user explicitly asks for a global install.
- Before querying, adding, modifying, deleting, updating, or rebuilding knowledge, run the project-local gate from this repository root: `.\audit-local.bat`.
- The project must remain operable from this directory after packaging. Do not require any external project folder for normal setup, audit, indexing, querying, Web UI, or packaging operations.
