# Testing and API documentation roadmap

Playwright browser tests and a Bruno health request are available. The additions below remain planned.

## Browser testing

Playwright runs against a locally started app in CI, covering navigation, desktop and mobile Chromium, story scrolling, keyboard interaction, and reduced motion. Add Firefox and WebKit after the initial suite is stable. Save failure traces without including secrets or private media. Keep visual and motion review as a manual check; automated tests cannot judge creative quality.

## API collection

The version-controlled Bruno collection starts with the health endpoint. Use a local base URL and environment variables for credentials; never commit secrets or real transcripts. Keep examples aligned with the FastAPI OpenAPI schema and pytest API tests. Add requests alongside each new endpoint.

## Working practice

For UI changes, check desktop and mobile, navigation, keyboard access, reduced motion and the changed interactions in a browser. Run relevant automated checks and report cases that fail or cannot be run. Update setup and developer documentation when behavior changes, and distinguish available features from proposals. Track substantive changes in a GitHub issue and link the pull request.

For each change, consider normal and failure paths, invalid and boundary inputs, retries and interruptions, resource limits, and recovery. Prioritize cases relevant to the change; do not claim exhaustive coverage. Record untested cases and blockers.
