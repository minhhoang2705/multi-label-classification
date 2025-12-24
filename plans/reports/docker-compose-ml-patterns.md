# Docker Compose Best Practices for Local ML API Deployment (2025)

## 1. Service Orchestration

### Base Configuration Pattern
- Base `docker-compose.yml`: production-ready shared config
- `docker-compose.override.yml`: auto-loaded dev overrides (hot-reload, bind mounts)
- `docker-compose.prod.yml`: production-specific overrides

```yaml
# docker-compose.yml (base)
services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: ml-api:latest
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    volumes:
      - models:/app/models
      - logs:/app/logs
    networks:
      - ml-network
    user: "1000:1000"  # non-root user

volumes:
  models:
    name: ml_models_v1
  logs:
    name: ml_logs

networks:
  ml-network:
    driver: bridge
```

### Optional Monitoring Stack
```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_admin_password
    secrets:
      - grafana_admin_password
```

## 2. Environment Variable Management

### Development (.env file)
```env
# .env (gitignored)
API_PORT=8000
MODEL_PATH=/app/models
LOG_LEVEL=DEBUG
WORKERS=2
```

### Production (Docker Secrets)
```yaml
# docker-compose.prod.yml
services:
  ml-api:
    environment:
      - LOG_LEVEL=INFO
      - WORKERS=4
    secrets:
      - api_key
      - db_password

secrets:
  api_key:
    file: ./secrets/api_key.txt  # managed externally
  db_password:
    environment: DB_PASSWORD  # from CI/CD
```

**Key Rules:**
- Never hardcode secrets in compose files
- `.env` files → gitignored, dev only
- Production → Docker secrets (`/run/secrets/<name>`) or external vaults (Vault, AWS Secrets Manager)
- Secrets mounted as in-memory files, never persisted to disk

## 3. Volume Strategies

### Named Volumes (Recommended for ML Models)
```yaml
volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/ml-models  # host path for large model files

  logs:
    driver: local  # Docker-managed, portable

  cache:
    driver: local
    name: model_cache_${MODEL_VERSION}
```

### Bind Mounts (Development Only)
```yaml
# docker-compose.override.yml
services:
  ml-api:
    volumes:
      - ./src:/app/src:ro  # read-only code hot-reload
      - ./tests:/app/tests
```

**Best Practices:**
- Named volumes for models/data (portable, managed backups)
- Descriptive names: `bert_models_v2` not `data`
- Bind mounts only in dev (remove in production)
- Regular cleanup: `docker volume prune`

## 4. Health Checks & Restart Policies

### Comprehensive Health Check
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
  interval: 30s        # check every 30s
  timeout: 10s         # 10s timeout per check
  retries: 3           # 3 failures → unhealthy
  start_period: 60s    # grace period for model loading
```

### Restart Policies
- `restart: always` → production (auto-recover from crashes)
- `restart: unless-stopped` → dev/optional services
- `restart: on-failure:3` → retry 3 times, then stop

**ML-Specific Considerations:**
- `start_period` ≥ model loading time (LLMs: 60-120s)
- Health endpoint should verify model readiness, not just API liveness

## 5. Port Mapping & Networking

### Bridge Network (Default, Recommended)
```yaml
networks:
  ml-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16

services:
  ml-api:
    networks:
      ml-network:
        ipv4_address: 172.28.0.10
    ports:
      - "127.0.0.1:8000:8000"  # localhost only
```

### Host Network (GPU Access)
```yaml
# For CUDA/GPU workloads requiring host network access
services:
  ml-api:
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Patterns:**
- Bridge: isolation, multi-service orchestration
- Host: GPU access, low-latency requirements
- Bind to `127.0.0.1:PORT` to prevent external exposure

## 6. Resource Limits (ML Workloads)

### CPU & Memory Constraints
```yaml
services:
  ml-api:
    deploy:
      resources:
        limits:
          cpus: '4.0'           # max 4 CPU cores
          memory: 16G           # hard limit 16GB
        reservations:
          cpus: '2.0'           # guaranteed 2 cores
          memory: 8G            # guaranteed 8GB
    environment:
      - OMP_NUM_THREADS=4       # limit inference parallelism
      - OPENBLAS_NUM_THREADS=4
```

### GPU Resource Management
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']     # specific GPU
          capabilities: [gpu]
    limits:
      memory: 24G               # prevent OOM from model + batch
```

**ML-Specific Limits:**
- Memory: model size + batch inference buffer (2-3x model size)
- CPU: match workers/threads to reserved cores
- GPU: use `device_ids` for multi-GPU isolation
- By default, containers have no limits → kernel scheduler decides (dangerous for ML)

## 7. Development vs Production Patterns

### docker-compose.override.yml (Dev, Auto-Loaded)
```yaml
services:
  ml-api:
    build:
      target: development
    volumes:
      - ./src:/app/src
    environment:
      - DEBUG=true
      - RELOAD=true
    ports:
      - "8000:8000"
      - "5678:5678"  # debugger port
```

### docker-compose.prod.yml (Production)
```yaml
services:
  ml-api:
    build:
      target: production
      args:
        - MODEL_VERSION=${MODEL_VERSION}
    environment:
      - DEBUG=false
      - WORKERS=8
    # No bind mounts (code inside container)
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4'
          memory: 16G
```

### Deployment Commands
```bash
# Development (auto-loads override)
docker compose up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Cleanup
docker compose down -v  # remove volumes too
```

## Key Takeaways

1. **Orchestration**: Base config production-ready, overrides for env-specific changes
2. **Secrets**: Docker secrets for prod, .env for dev (gitignored)
3. **Volumes**: Named volumes for models (portable), bind mounts dev-only
4. **Health**: ML-aware checks (model ready, not just API alive), `start_period` ≥ load time
5. **Networking**: Bridge default (isolation), host for GPU, bind localhost only
6. **Resources**: Hard limits critical (2-3x model size for memory), match CPU to workers
7. **Override Pattern**: `docker-compose.override.yml` auto-loaded, explicit `-f` for prod

## Unresolved Questions
- None

## Sources
- [Use Compose in production | Docker Docs](https://docs.docker.com/compose/how-tos/production/)
- [Secrets in Compose | Docker Docs](https://docs.docker.com/compose/how-tos/use-secrets/)
- [Resource constraints | Docker Docs](https://docs.docker.com/engine/containers/resource_constraints/)
- [Volumes | Docker Docs](https://docs.docker.com/engine/storage/volumes/)
- [Best Practices Around Production Ready Web Apps with Docker Compose — Nick Janetakis](https://nickjanetakis.com/blog/best-practices-around-production-ready-web-apps-with-docker-compose)
