import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import os
from pathlib import Path
import threading
import time
import hashlib
import shutil

logger = logging.getLogger(__name__)

class ModelPersistenceManager:
    """
    Advanced Model Persistence Manager for AI Models
    Features:
    - Multi-format model storage (JSON, Pickle, HDF5)
    - Version control and model tracking
    - Automatic backup and recovery
    - Model metadata management
    - Performance monitoring
    """
    
    def __init__(self, storage_dir: str = "model_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Storage structure
        self.models_dir = self.storage_dir / "models"
        self.backups_dir = self.storage_dir / "backups"
        self.metadata_dir = self.storage_dir / "metadata"
        
        # Create directories
        for dir_path in [self.models_dir, self.backups_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.model_versions = {}
        self.model_metadata = {}
        
        # Performance tracking
        self.save_times = []
        self.load_times = []
        self.backup_times = []
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Model Persistence Manager initialized at {self.storage_dir}")
    
    def _load_registry(self):
        """Load existing model registry"""
        try:
            registry_file = self.storage_dir / "model_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    self.model_registry = data.get('models', {})
                    self.model_versions = data.get('versions', {})
                    self.model_metadata = data.get('metadata', {})
                
                logger.info(f"Loaded registry with {len(self.model_registry)} models")
            else:
                logger.info("No existing registry found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            registry_file = self.storage_dir / "model_registry.json"
            data = {
                'models': self.model_registry,
                'versions': self.model_versions,
                'metadata': self.model_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Model registry saved")
            
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def save_model(self, model_name: str, model_data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Save a model with metadata"""
        try:
            start_time = time.time()
            
            with self.lock:
                # Generate model hash
                model_hash = self._generate_model_hash(model_data)
                
                # Create version
                version = self._get_next_version(model_name)
                
                # Prepare metadata
                model_metadata = {
                    'name': model_name,
                    'version': version,
                    'hash': model_hash,
                    'created_at': datetime.now().isoformat(),
                    'size_bytes': self._estimate_model_size(model_data),
                    'format': self._determine_format(model_data),
                    'custom_metadata': metadata or {}
                }
                
                # Save model data
                success = self._save_model_data(model_name, version, model_data, model_metadata['format'])
                
                if success:
                    # Update registry
                    if model_name not in self.model_registry:
                        self.model_registry[model_name] = []
                    
                    self.model_registry[model_name].append({
                        'version': version,
                        'hash': model_hash,
                        'created_at': model_metadata['created_at'],
                        'format': model_metadata['format']
                    })
                    
                    # Store metadata
                    self.model_metadata[f"{model_name}_v{version}"] = model_metadata
                    
                    # Update versions
                    self.model_versions[model_name] = version
                    
                    # Save registry
                    self._save_registry()
                    
                    # Create backup
                    self._create_backup(model_name, version)
                    
                    save_time = time.time() - start_time
                    self.save_times.append(save_time)
                    
                    logger.info(f"Saved model {model_name} v{version} in {save_time:.2f}s")
                    return True
                else:
                    logger.error(f"Failed to save model data for {model_name}")
                    return False
                
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def _save_model_data(self, model_name: str, version: int, model_data: Any, format_type: str) -> bool:
        """Save model data in specified format"""
        try:
            model_file = self.models_dir / f"{model_name}_v{version}.{format_type}"
            
            if format_type == "json":
                # Save as JSON (for simple data structures)
                if isinstance(model_data, (dict, list, str, int, float, bool)) or model_data is None:
                    with open(model_file, 'w') as f:
                        json.dump(model_data, f, indent=2)
                else:
                    # Convert to serializable format
                    serializable_data = self._convert_to_serializable(model_data)
                    with open(model_file, 'w') as f:
                        json.dump(serializable_data, f, indent=2)
            
            elif format_type == "pkl":
                # Save as pickle
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
            
            elif format_type == "npz":
                # Save as numpy compressed format
                if isinstance(model_data, np.ndarray):
                    np.savez_compressed(model_file, data=model_data)
                else:
                    # Convert to numpy array
                    np_data = np.array(model_data)
                    np.savez_compressed(model_file, data=np_data)
            
            else:
                # Default to JSON
                serializable_data = self._convert_to_serializable(model_data)
                with open(model_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model data: {e}")
            return False
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return str(obj)
        except Exception as e:
            logger.error(f"Error converting to serializable: {e}")
            return str(obj)
    
    def load_model(self, model_name: str, version: Optional[int] = None) -> Optional[Any]:
        """Load a model by name and version"""
        try:
            start_time = time.time()
            
            with self.lock:
                # Get version
                if version is None:
                    version = self.model_versions.get(model_name)
                    if version is None:
                        logger.error(f"No version found for model {model_name}")
                        return None
                
                # Check if model exists
                if model_name not in self.model_registry:
                    logger.error(f"Model {model_name} not found in registry")
                    return None
                
                # Find model in registry
                model_info = None
                for model in self.model_registry[model_name]:
                    if model['version'] == version:
                        model_info = model
                        break
                
                if model_info is None:
                    logger.error(f"Version {version} not found for model {model_name}")
                    return None
                
                # Load model data
                model_data = self._load_model_data(model_name, version, model_info['format'])
                
                if model_data is not None:
                    load_time = time.time() - start_time
                    self.load_times.append(load_time)
                    
                    logger.info(f"Loaded model {model_name} v{version} in {load_time:.2f}s")
                    return model_data
                else:
                    logger.error(f"Failed to load model data for {model_name} v{version}")
                    return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_model_data(self, model_name: str, version: int, format_type: str) -> Optional[Any]:
        """Load model data from file"""
        try:
            model_file = self.models_dir / f"{model_name}_v{version}.{format_type}"
            
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return None
            
            if format_type == "json":
                with open(model_file, 'r') as f:
                    return json.load(f)
            
            elif format_type == "pkl":
                with open(model_file, 'rb') as f:
                    return pickle.load(f)
            
            elif format_type == "npz":
                data = np.load(model_file)
                return data['data']
            
            else:
                # Try JSON as fallback
                with open(model_file, 'r') as f:
                    return json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading model data: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata"""
        try:
            models = []
            
            for model_name, versions in self.model_registry.items():
                latest_version = self.model_versions.get(model_name, 0)
                latest_metadata = self.model_metadata.get(f"{model_name}_v{latest_version}", {})
                
                models.append({
                    'name': model_name,
                    'latest_version': latest_version,
                    'total_versions': len(versions),
                    'created_at': latest_metadata.get('created_at'),
                    'size_bytes': latest_metadata.get('size_bytes', 0),
                    'format': latest_metadata.get('format', 'unknown')
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model"""
        try:
            if model_name not in self.model_registry:
                return []
            
            versions = []
            for model_info in self.model_registry[model_name]:
                version = model_info['version']
                metadata = self.model_metadata.get(f"{model_name}_v{version}", {})
                
                versions.append({
                    'version': version,
                    'hash': model_info['hash'],
                    'created_at': model_info['created_at'],
                    'format': model_info['format'],
                    'size_bytes': metadata.get('size_bytes', 0),
                    'custom_metadata': metadata.get('custom_metadata', {})
                })
            
            # Sort by version
            versions.sort(key=lambda x: x['version'], reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"Error getting versions for {model_name}: {e}")
            return []
    
    def delete_model(self, model_name: str, version: Optional[int] = None) -> bool:
        """Delete a model or specific version"""
        try:
            with self.lock:
                if model_name not in self.model_registry:
                    logger.error(f"Model {model_name} not found")
                    return False
                
                if version is None:
                    # Delete all versions
                    versions_to_delete = [model['version'] for model in self.model_registry[model_name]]
                    
                    for v in versions_to_delete:
                        self._delete_model_version(model_name, v)
                    
                    # Remove from registry
                    del self.model_registry[model_name]
                    if model_name in self.model_versions:
                        del self.model_versions[model_name]
                    
                    logger.info(f"Deleted all versions of model {model_name}")
                    
                else:
                    # Delete specific version
                    success = self._delete_model_version(model_name, version)
                    if success:
                        # Remove from registry
                        self.model_registry[model_name] = [
                            model for model in self.model_registry[model_name] 
                            if model['version'] != version
                        ]
                        
                        # Update latest version if needed
                        if self.model_versions.get(model_name) == version:
                            if self.model_registry[model_name]:
                                self.model_versions[model_name] = max(
                                    model['version'] for model in self.model_registry[model_name]
                                )
                            else:
                                del self.model_versions[model_name]
                        
                        logger.info(f"Deleted model {model_name} v{version}")
                    else:
                        return False
                
                # Save registry
                self._save_registry()
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def _delete_model_version(self, model_name: str, version: int) -> bool:
        """Delete a specific model version"""
        try:
            # Get model info
            model_info = None
            for model in self.model_registry[model_name]:
                if model['version'] == version:
                    model_info = model
                    break
            
            if model_info is None:
                return False
            
            # Delete model file
            model_file = self.models_dir / f"{model_name}_v{version}.{model_info['format']}"
            if model_file.exists():
                model_file.unlink()
            
            # Delete metadata
            metadata_key = f"{model_name}_v{version}"
            if metadata_key in self.model_metadata:
                del self.model_metadata[metadata_key]
            
            # Delete backup
            backup_file = self.backups_dir / f"{model_name}_v{version}.{model_info['format']}"
            if backup_file.exists():
                backup_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model version {model_name} v{version}: {e}")
            return False
    
    def _generate_model_hash(self, model_data: Any) -> str:
        """Generate hash for model data"""
        try:
            # Convert to string representation
            if isinstance(model_data, (dict, list)):
                data_str = json.dumps(model_data, sort_keys=True)
            else:
                data_str = str(model_data)
            
            # Generate hash
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Error generating model hash: {e}")
            return "unknown"
    
    def _get_next_version(self, model_name: str) -> int:
        """Get next version number for model"""
        current_version = self.model_versions.get(model_name, 0)
        return current_version + 1
    
    def _estimate_model_size(self, model_data: Any) -> int:
        """Estimate size of model data in bytes"""
        try:
            if isinstance(model_data, (dict, list)):
                return len(json.dumps(model_data).encode())
            elif isinstance(model_data, np.ndarray):
                return model_data.nbytes
            elif isinstance(model_data, pd.DataFrame):
                return model_data.memory_usage(deep=True).sum()
            else:
                return len(str(model_data).encode())
        except Exception as e:
            logger.error(f"Error estimating model size: {e}")
            return 0
    
    def _determine_format(self, model_data: Any) -> str:
        """Determine best format for model data"""
        try:
            if isinstance(model_data, np.ndarray):
                return "npz"
            elif isinstance(model_data, (dict, list, str, int, float, bool)) or model_data is None:
                return "json"
            else:
                return "pkl"
        except Exception as e:
            logger.error(f"Error determining format: {e}")
            return "json"
    
    def _create_backup(self, model_name: str, version: int):
        """Create backup of model"""
        try:
            start_time = time.time()
            
            # Find model file
            model_files = list(self.models_dir.glob(f"{model_name}_v{version}.*"))
            if not model_files:
                return
            
            model_file = model_files[0]
            backup_file = self.backups_dir / model_file.name
            
            # Copy file
            shutil.copy2(model_file, backup_file)
            
            backup_time = time.time() - start_time
            self.backup_times.append(backup_time)
            
            logger.debug(f"Created backup of {model_name} v{version} in {backup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def restore_from_backup(self, model_name: str, version: int) -> bool:
        """Restore model from backup"""
        try:
            # Find backup file
            backup_files = list(self.backups_dir.glob(f"{model_name}_v{version}.*"))
            if not backup_files:
                logger.error(f"No backup found for {model_name} v{version}")
                return False
            
            backup_file = backup_files[0]
            model_file = self.models_dir / backup_file.name
            
            # Copy from backup
            shutil.copy2(backup_file, model_file)
            
            logger.info(f"Restored {model_name} v{version} from backup")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            return {
                'total_models': len(self.model_registry),
                'total_versions': sum(len(versions) for versions in self.model_registry.values()),
                'total_size_bytes': sum(
                    self.model_metadata.get(f"{model_name}_v{self.model_versions[model_name]}", {}).get('size_bytes', 0)
                    for model_name in self.model_versions.keys()
                ),
                'avg_save_time': np.mean(self.save_times) if self.save_times else 0,
                'avg_load_time': np.mean(self.load_times) if self.load_times else 0,
                'avg_backup_time': np.mean(self.backup_times) if self.backup_times else 0,
                'total_saves': len(self.save_times),
                'total_loads': len(self.load_times),
                'total_backups': len(self.backup_times)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                'total_models': 0,
                'total_versions': 0,
                'total_size_bytes': 0,
                'avg_save_time': 0,
                'avg_load_time': 0,
                'avg_backup_time': 0,
                'total_saves': 0,
                'total_loads': 0,
                'total_backups': 0
            }
    
    def cleanup_old_versions(self, model_name: str, keep_versions: int = 3) -> int:
        """Clean up old versions of a model, keeping only the latest N versions"""
        try:
            if model_name not in self.model_registry:
                return 0
            
            versions = [model['version'] for model in self.model_registry[model_name]]
            versions.sort(reverse=True)
            
            versions_to_delete = versions[keep_versions:]
            deleted_count = 0
            
            for version in versions_to_delete:
                if self.delete_model(model_name, version):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old versions of {model_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
            return 0
    
    def export_model(self, model_name: str, version: Optional[int] = None, 
                    export_path: str = None) -> Optional[str]:
        """Export model to external location"""
        try:
            # Load model
            model_data = self.load_model(model_name, version)
            if model_data is None:
                return None
            
            # Determine export path
            if export_path is None:
                export_path = f"{model_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Export model
            if isinstance(model_data, (dict, list, str, int, float, bool)) or model_data is None:
                with open(export_path, 'w') as f:
                    json.dump(model_data, f, indent=2)
            else:
                # Use pickle for complex objects
                with open(export_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"Exported model {model_name} to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return None

# Initialize global model persistence manager
model_persistence_manager = ModelPersistenceManager() 