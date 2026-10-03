---
name: purestorage
description: PureStorage FlashArray/FlashBlade provisioning, naming conventions, and REST API usage.
always_on: false
---

Volume names follow the pattern `<env>-<app>-<purpose>-<size>` (e.g. `prod-erp-data-500g`).

For FlashArray REST API calls, use the `purestorage_*` tools if configured in the REST allowlist. See reference/rest-api-cheatsheet.md for endpoint details.
