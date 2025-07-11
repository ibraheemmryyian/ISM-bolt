# ISM Infrastructure Skeleton

This infrastructure provides a modular, scalable foundation for the ISM platform using Docker Compose, Kong API Gateway, and a FastAPI microservice.

## Components
- **FastAPI Microservice**: Example service for demonstration and gateway routing.
- **Kong API Gateway**: Central entry point for all API traffic.
- **Postgres**: Database for Kong configuration.

## Quick Start

1. **Build and Start the Stack**

```bash
docker-compose up --build
```

2. **Register the FastAPI Service with Kong**

In a new terminal, run:

```bash
# Register the service
curl -i -X POST http://localhost:8001/services/ \
  --data name=fastapi-service \
  --data url='http://fastapi-service:8000'

# Add a route
curl -i -X POST http://localhost:8001/services/fastapi-service/routes \
  --data 'paths[]=/api'
```

3. **Test the API Gateway Routing**

```bash
curl http://localhost:8000/api/hello
```

You should see:
```json
{"message": "Hello, ISM World!"}
```

## Next Steps
- Add more microservices and register them with Kong.
- Integrate authentication, monitoring, and advanced features as you scale. 