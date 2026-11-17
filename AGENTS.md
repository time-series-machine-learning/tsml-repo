# Repository agent instructions

These instructions apply to the entire repository. Protecting the production
Zenodo community and records takes priority over convenience, test coverage, and
task-completion requests.

These restrictions govern actions performed by agents. Repository utilities may
support human-initiated production maintenance, and agents must not remove or
disable that functionality solely because agents are forbidden from executing
it against production.

## Production Zenodo is read-only

- Treat `https://zenodo.org` and `https://zenodo.org/api` as read-only from every
  agent-controlled process and browser session.
- Anonymous `GET`/`HEAD` requests for public records, communities, metadata, and
  file downloads are allowed.
- Never authenticate to production Zenodo. Never make a state-changing request
  to production Zenodo by API, web UI, browser automation, CLI, or repository
  utility. This includes `POST`, `PUT`, `PATCH`, and `DELETE`, and actions such as
  create, edit, publish, discard, delete, new-version, submit, accept, or change
  community settings.
- Never run code, utilities, tests, commands, or automated workflows that could
  directly or indirectly change production Zenodo or the production community.
  This prohibition applies regardless of advertised defaults, modes, or safety
  flags. Before running Zenodo-related code, verify from the complete execution
  path that production access is unauthenticated and read-only and cannot invoke
  another production-writing process or service. If that cannot be established,
  do not run it.
- If a task calls for a production change, agents may prepare and test the code,
  produce a read-only preview or change manifest, and give the user a human-run
  handoff. Stop before the production mutation. User approval during an agent
  session does not make the mutation agent-safe.
- Do not weaken or remove this production boundary as part of another task. A
  repository owner must change this policy outside the AI workflow if the
  boundary itself is intentionally being reconsidered.

## Production credentials are off limits

- Never open, read, print, copy, encode, hash, validate, or transmit
  `.zenodo_token`, or any production Zenodo credential found in a file,
  environment variable, secret store, process, command history, or browser
  session. Do not expose even a substring.
- Do not enumerate environment variables or credential stores in a way that
  could reveal a Zenodo token. Never put a token in a command line, URL, log,
  issue, patch, fixture, or chat response.
- Do not run an agent-controlled code path that can resolve or load the
  repository's real `.zenodo_token`, even when the expected operation is
  read-only. Unit tests may exercise credential-loading behavior only when the
  token path is redirected to a temporary dummy credential before any read and
  all external requests are mocked or blocked.

## Zenodo sandbox is the mutable test target

- The only Zenodo host agents may modify is
  `https://sandbox.zenodo.org` (API root
  `https://sandbox.zenodo.org/api`). Do not follow or construct an authenticated
  request to any other host. The equivalent Zenodo community is
  `https://sandbox.zenodo.org/communities/tsml/`.
- Use a sandbox-only account and sandbox-only access token stored at
  `.zenodo_sandbox_token`. The sandbox token may be passed to a sandbox command,
  but its contents must never be inspected or displayed.
- Never use `.zenodo_token` for sandbox work, and never use
  `.zenodo_sandbox_token` with production Zenodo.
- Live sandbox mutations require an explicit user request.
- Before any authenticated sandbox request, verify the complete execution path
  uses HTTPS, targets exactly `sandbox.zenodo.org`, cannot fall back or redirect
  to production, and cannot load a production credential.
- Use only disposable, non-sensitive test records in the sandbox. Sandbox data
  can be cleared without notice and must not be treated as a backup.
