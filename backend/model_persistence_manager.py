import os
import json
import pickle
import logging
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import threading
import time
import zipfile
import tempfile

logger = logging.getLogger(__name__)

class ModelPersistenceManager:
    """
    Centralized Model Persistence Manager for ISM AI Platform
    Features:
    - Unified model storage and loading
    - Version control and rollback
    - Model metadata tracking
    - Automatic backup and recovery
    - Model performance monitoring
    - Cross-platform compatibility
    """
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.model_metadata = {}
        self.model_versions = {}
        
        # Performance tracking
        self.performance_history = {}
        self.load_times = {}
        
        # Backup configuration
        self.backup_config = {
            'auto_backup': True,
            'backup_interval_hours': 24,
            'max_backups': 10,
            'backup_dir': self.base_dir / "backups"
        }
        self.backup_config['backup_dir'].mkdir(exist_ok=True)
        
        # Threading
        self.lock = threading.Lock()
        
        # Load existing registry
        self._load_registry()
        
        # Start backup scheduler
        self._start_backup_scheduler()
        
        logger.info(f"Model Persistence Manager initialized at {self.base_dir}")

    def _load_registry(self):
        """Load existing model registry"""
        try:
            registry_path = self.base_dir / "model_registry.json"
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Loaded {len(self.model_registry)} models from registry")
            
            metadata_path = self.base_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
                    
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")

    def _save_registry(self):
        """Save model registry to disk"""
        try:
            with self.lock:
                registry_path = self.base_dir / "model_registry.json"
                with open(registry_path, 'w') as f:
                    json.dump(self.model_registry, f, indent=2)
                
                metadata_path = self.base_dir / "model_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(self.model_metadata, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def save_model(self, model_name: str, model: Any, metadata: Dict[str, Any] = None) -> bool:
        """Save a model with metadata"""
        try:
            with self.lock:
                start_time = time.time()
                
                # Create model directory
                model_dir = self.base_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                # Generate version
                version = self._generate_version(model_name)
                
                # Create version directory
                version_dir = model_dir / f"v{version}"
                version_dir.mkdir(exist_ok=True)
                
                # Save model
                model_path = version_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Prepare metadata
                if metadata is None:
                    metadata = {}
                
                model_metadata = {
                    'name': model_name,
                    'version': version,
                    'created_at': datetime.now().isoformat(),
                    'model_type': type(model).__name__,
                    'file_size': model_path.stat().st_size,
                    'checksum': self._calculate_checksum(model_path),
                    **metadata
                }
                
                # Save metadata
                metadata_path = version_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Update registry
                if model_name not in self.model_registry:
                    self.model_registry[model_name] = {
                        'current_version': version,
                        'versions': [],
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat()
                    }
                
                self.model_registry[model_name]['current_version'] = version
                self.model_registry[model_name]['last_updated'] = datetime.now().isoformat()
                
                if version not in self.model_registry[model_name]['versions']:
                    self.model_registry[model_name]['versions'].append(version)
                
                # Store metadata
                self.model_metadata[f"{model_name}_v{version}"] = model_metadata
                
                # Save registry
                self._save_registry()
                
                save_time = time.time() - start_time
                logger.info(f"Saved model {model_name} v{version} in {save_time:.2f}s")
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False

    def load_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load a model by name and version"""
        try:
            with self.lock:
                start_time = time.time()
                
                # Get version to load
                if version is None:
                    if model_name not in self.model_registry:
                        logger.error(f"Model {model_name} not found in registry")
                        return None
                    version = self.model_registry[model_name]['current_version']
                
                # Construct path
                model_path = self.base_dir / model_name / f"v{version}" / "model.pkl"
                
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None
                
                # Load model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                load_time = time.time() - start_time
                self.load_times[f"{model_name}_v{version}"] = load_time
                
                logger.info(f"Loaded model {model_name} v{version} in {load_time:.2f}s")
                
                return model
                
        except Exception as e:
            logger.error(f"Error loading model {model_name} v{version}: {e}")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            models = []
            for model_name, info in self.model_registry.items():
                current_version = info['current_version']
                metadata_key = f"{model_name}_v{current_version}"
                metadata = self.model_metadata.get(metadata_key, {})
                
                models.append({
                    'name': model_name,
                    'current_version': current_version,
                    'versions': info['versions'],
                    'created_at': info['created_at'],
                    'last_updated': info['last_updated'],
                    'model_type': metadata.get('model_type', 'Unknown'),
                    'file_size': metadata.get('file_size', 0)
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model version"""
        try:
            if version is None:
                if model_name not in self.model_registry:
                    return None
                version = self.model_registry[model_name]['current_version']
            
            metadata_key = f"{model_name}_v{version}"
            return self.model_metadata.get(metadata_key)
            
        except Exception as e:
            logger.error(f"Error getting metadata for {model_name} v{version}: {e}")
            return None

    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model version"""
        try:
            with self.lock:
                if version is None:
                    if model_name not in self.model_registry:
                        return False
                    version = self.model_registry[model_name]['current_version']
                
                # Remove from registry
                if model_name in self.model_registry:
                    if version in self.model_registry[model_name]['versions']:
                        self.model_registry[model_name]['versions'].remove(version)
                    
                    # If no versions left, remove model entirely
                    if not self.model_registry[model_name]['versions']:
                        del self.model_registry[model_name]
                    else:
                        # Update current version if needed
                        if self.model_registry[model_name]['current_version'] == version:
                            self.model_registry[model_name]['current_version'] = self.model_registry[model_name]['versions'][-1]
                
                # Remove metadata
                metadata_key = f"{model_name}_v{version}"
                if metadata_key in self.model_metadata:
                    del self.model_metadata[metadata_key]
                
                # Remove files
                model_dir = self.base_dir / model_name / f"v{version}"
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Deleted model {model_name} v{version}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name} v{version}: {e}")
            return False

    def rollback_model(self, model_name: str, version: str) -> bool:
        """Rollback to a specific model version"""
        try:
            with self.lock:
                if model_name not in self.model_registry:
                    logger.error(f"Model {model_name} not found")
                    return False
                
                if version not in self.model_registry[model_name]['versions']:
                    logger.error(f"Version {version} not found for model {model_name}")
                    return False
                
                # Update current version
                self.model_registry[model_name]['current_version'] = version
                self.model_registry[model_name]['last_updated'] = datetime.now().isoformat()
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Rolled back model {model_name} to version {version}")
                return True
                
        except Exception as e:
            logger.error(f"Error rolling back model {model_name} to v{version}: {e}")
            return False

    def create_backup(self) -> bool:
        """Create a backup of all models"""
        try:
            with self.lock:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_config['backup_dir'] / f"backup_{timestamp}.zip"
                
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add all model files
                    for model_name in self.model_registry.keys():
                        model_dir = self.base_dir / model_name
                        if model_dir.exists():
                            for root, dirs, files in os.walk(model_dir):
                                for file in files:
                                    file_path = Path(root) / file
                                    arc_name = file_path.relative_to(self.base_dir)
                                    zipf.write(file_path, arc_name)
                    
                    # Add registry files
                    registry_path = self.base_dir / "model_registry.json"
                    if registry_path.exists():
                        zipf.write(registry_path, "model_registry.json")
                    
                    metadata_path = self.base_dir / "model_metadata.json"
                    if metadata_path.exists():
                        zipf.write(metadata_path, "model_metadata.json")
                
                # Clean old backups
                self._cleanup_old_backups()
                
                logger.info(f"Created backup: {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def restore_backup(self, backup_path: str) -> bool:
        """Restore from a backup file"""
        try:
            with self.lock:
                backup_file = Path(backup_path)
                if not backup_file.exists():
                    logger.error(f"Backup file not found: {backup_path}")
                    return False
                
                # Create temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Extract backup
                    with zipfile.ZipFile(backup_file, 'r') as zipf:
                        zipf.extractall(temp_path)
                    
                    # Restore files
                    for item in temp_path.iterdir():
                        if item.is_file():
                            # Registry files
                            if item.name in ['model_registry.json', 'model_metadata.json']]:
                                shutil.copy2(item, self.base_dir / item.name)
                        elif item.is_dir():
                            # Model directories
                            shutil.copytree(item, self.base_dir / item.name, dirs_exist_ok=True)
                
                # Reload registry
                self._load_registry()
                
                logger.info(f"Restored from backup: {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False

    def _generate_version(self, model_name: str) -> str:
        """Generate a new version number for a model"""
        if model_name not in self.model_registry:
            return "1.0.0"
        
        current_version = self.model_registry[model_name]['current_version']
        
        # Simple version increment
        try:
            major, minor, patch = map(int, current_version.split('.'))
            patch += 1
            return f"{major}.{minor}.{patch}"
        except:
            return f"{current_version}.1"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _cleanup_old_backups(self):
        """Remove old backup files"""
        try:
            backup_files = list(self.backup_config['backup_dir'].glob("backup_*.zip"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            for backup_file in backup_files[self.backup_config['max_backups']:]:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

    def _start_backup_scheduler(self):
        """Start automatic backup scheduler"""
        if not self.backup_config['auto_backup']:
            return
        
        def backup_scheduler():
            while True:
                try:
                    time.sleep(self.backup_config['backup_interval_hours'] * 3600)
                    self.create_backup()
                except Exception as e:
                    logger.error(f"Error in backup scheduler: {e}")
        
        # Start backup thread
        backup_thread = threading.Thread(target=backup_scheduler, daemon=True)
        backup_thread.start()
        logger.info("Backup scheduler started")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = {
                'total_models': len(self.model_registry),
                'total_versions': sum(len(info['versions']) for info in self.model_registry.values()),
                'average_load_time': np.mean(list(self.load_times.values())) if self.load_times else 0,
                'total_backups': len(list(self.backup_config['backup_dir'].glob("backup_*.zip"))),
                'registry_size': len(self.model_registry),
                'metadata_size': len(self.model_metadata)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

    def validate_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Validate a model's integrity"""
        try:
            if version is None:
                if model_name not in self.model_registry:
                    return False
                version = self.model_registry[model_name]['current_version']
            
            # Check if files exist
            model_path = self.base_dir / model_name / f"v{version}" / "model.pkl"
            metadata_path = self.base_dir / model_name / f"v{version}" / "metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                return False
            
            # Check checksum
            metadata = self.get_model_metadata(model_name, version)
            if metadata:
                current_checksum = self._calculate_checksum(model_path)
                stored_checksum = metadata.get('checksum')
                if current_checksum != stored_checksum:
                    logger.warning(f"Checksum mismatch for {model_name} v{version}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model {model_name} v{version}: {e}")
            return False

    def export_model(self, model_name: str, version: Optional[str] = None, 
                    export_path: Optional[str] = None) -> Optional[str]:
        """Export a model to a portable format"""
        try:
            if version is None:
                if model_name not in self.model_registry:
                    return None
                version = self.model_registry[model_name]['current_version']
            
            # Load model
            model = self.load_model(model_name, version)
            if model is None:
                return None
            
            # Get metadata
            metadata = self.get_model_metadata(model_name, version)
            
            # Create export package
            if export_path is None:
                export_path = f"{model_name}_v{version}_export.zip"
            
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add model
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    pickle.dump(model, tmp)
                    tmp_path = tmp.name
                
                zipf.write(tmp_path, "model.pkl")
                os.unlink(tmp_path)
                
                # Add metadata
                if metadata:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                        json.dump(metadata, tmp, indent=2)
                        tmp_path = tmp.name
                    
                    zipf.write(tmp_path, "metadata.json")
                    os.unlink(tmp_path)
            
            logger.info(f"Exported model {model_name} v{version} to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting model {model_name} v{version}: {e}")
            return None

    def import_model(self, import_path: str, model_name: Optional[str] = None) -> bool:
        """Import a model from an export package"""
        try:
            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Extract model
                model_data = zipf.read("model.pkl")
                model = pickle.loads(model_data)
                
                # Extract metadata
                metadata = {}
                if "metadata.json" in zipf.namelist():
                    metadata_data = zipf.read("metadata.json")
                    metadata = json.loads(metadata_data)
                
                # Determine model name
                if model_name is None:
                    model_name = metadata.get('name', f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Save model
                return self.save_model(model_name, model, metadata)
                
        except Exception as e:
            logger.error(f"Error importing model from {import_path}: {e}")
            return False

# Initialize global model persistence manager
model_persistence_manager = ModelPersistenceManager() 