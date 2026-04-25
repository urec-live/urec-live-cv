# DB Update Flow — CV Pipeline to Spring Boot

## With `--post-url` only (FastAPI)

```
Camera → CV pipeline → POST /equipment/status → FastAPI (in-memory store)
                                                      ↑
Spring Boot ←── polls every few seconds ─────────────┘
     ↓
WebSocket broadcast → all connected clients
```

Spring Boot's `CvAvailabilityService` periodically calls `GET /equipment/status` on the FastAPI server and pushes the result into its own WebSocket topic. The DB (`equipment` table) is **not** written — only the live broadcast is affected.

## With `--springboot-url` added

```
Camera → CV pipeline ──→ POST /equipment/status → FastAPI (in-memory)
                    └──→ PUT /api/machines/code/{id}/status → Spring Boot
                                                                    ↓
                                                          DB write + WebSocket broadcast
```

The `PUT /api/machines/code/{id}/status` endpoint (`MachineController`) writes the new status to PostgreSQL immediately and also triggers the WebSocket broadcast. This is the same endpoint the QR scanner uses for check-in/out.

## Summary

| Flag | FastAPI updated | DB written | Real-time broadcast |
|---|---|---|---|
| `--post-url` only | Yes | No (polling lag) | Yes (after next poll) |
| `--post-url` + `--springboot-url` | Yes | Yes (immediate) | Yes (immediate) |
| `--springboot-url` only | No | Yes (immediate) | Yes (immediate) |

## When to use each mode

- **`--post-url` only** — dev/demo; status visible in frontend but not persisted across backend restarts
- **`--post-url` + `--springboot-url`** — production; FastAPI acts as a cache, Spring Boot writes to DB and broadcasts immediately
- **`--springboot-url` only** — skips FastAPI entirely; Spring Boot is the single source of truth

## Spring Boot configuration (polling mode)

In `urec-live-backend/src/main/resources/application.properties`:

```properties
APP_CV_SECONDARY_ENABLED=true
APP_CV_SECONDARY_BASE_URL=http://localhost:8000
APP_CV_SECONDARY_CONFIDENCE_THRESHOLD=0.65
APP_CV_SECONDARY_CACHE_TTL_MS=3000
```
