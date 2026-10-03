# PureStorage FlashArray REST API quick reference

- `GET /api/2.x/volumes` — list volumes
- `POST /api/2.x/volumes` — create a volume, body: `{"names": ["..."], "provisioned": <bytes>}`
- `DELETE /api/2.x/volumes/<name>` — destroy (soft-delete, recoverable for a period)
