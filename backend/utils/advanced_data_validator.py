import json
import logging
from typing import Dict, Any, List, Optional
from jsonschema import validate, ValidationError
from utils.distributed_logger import DistributedLogger

class AdvancedDataValidator:
    """Distributed, schema-based, and ML-specific data validator"""
    def __init__(self, schema: Optional[Dict[str, Any]] = None, logger: Optional[DistributedLogger] = None):
        self.schema = schema
        self.logger = logger or DistributedLogger('AdvancedDataValidator')
    def set_schema(self, schema: Dict[str, Any]):
        self.schema = schema
    def validate(self, data: Any) -> bool:
        if self.schema is None:
            self.logger.warning("No schema set for validation.")
            return False
        try:
            validate(instance=data, schema=self.schema)
            self.logger.info("Data validation passed.")
            return True
        except ValidationError as e:
            self.logger.error(f"Data validation failed: {e.message}")
            return False
    def validate_batch(self, batch: List[Any]) -> List[bool]:
        return [self.validate(item) for item in batch]
    def validate_ml_input(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        missing = [f for f in required_fields if f not in data]
        if missing:
            self.logger.error(f"Missing required fields: {missing}")
            return False
        self.logger.info("ML input validation passed.")
        return True
    def validate_types(self, data: Dict[str, Any], type_map: Dict[str, type]) -> bool:
        for k, t in type_map.items():
            if k in data and not isinstance(data[k], t):
                self.logger.error(f"Field {k} has wrong type: expected {t}, got {type(data[k])}")
                return False
        self.logger.info("Type validation passed.")
        return True 