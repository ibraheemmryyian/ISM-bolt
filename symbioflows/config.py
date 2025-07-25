import os, pathlib, functools, yaml

_REGISTRY_FILE = pathlib.Path(os.getenv("SERVICE_REGISTRY_PATH", "config/service_registry.yaml"))

@functools.lru_cache(maxsize=1)
def load_service_registry():
    if not _REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Service registry file not found: {_REGISTRY_FILE}")
    with _REGISTRY_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def get_service_endpoint(service_id: str):
    registry = load_service_registry()
    if service_id not in registry:
        raise KeyError(service_id)
    info = registry[service_id]
    return info.get("host", "localhost"), int(info["port"])