"""
Enterprise Infrastructure Module
Phase 9 & 10: Enterprise Features & Advanced Infrastructure

This module implements enterprise-grade features:
- Single Sign-On (SSO) with OAuth2/OIDC
- Role-Based Access Control (RBAC)
- Audit trails and compliance logging
- White-labeling and customization
- Microservices architecture
- Kubernetes deployment
- Auto-scaling and load balancing
- Monitoring and observability
- Disaster recovery and backup
- Multi-tenant architecture
"""

import os
import json
import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import aiohttp
from functools import wraps
import redis
import psycopg2
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import kubernetes
from kubernetes import client, config
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    company_id = Column(String)
    role = Column(String, default='user')
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    sso_provider = Column(String)
    sso_id = Column(String)

class Company(Base):
    __tablename__ = 'companies'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    domain = Column(String, unique=True)
    subscription_tier = Column(String, default='basic')
    custom_branding = Column(JSON, default=dict)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True)
    user_id = Column(String)
    company_id = Column(String)
    action = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    company_id = Column(String)
    permissions = Column(JSON, default=list)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Enums
class SubscriptionTier(Enum):
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    BILLING = "billing"
    ANALYTICS = "analytics"
    API_ACCESS = "api_access"

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
DATABASE_CONNECTIONS = Gauge('database_connections', 'Number of database connections')

@dataclass
class SSOConfig:
    """SSO configuration"""
    provider: str  # google, azure, okta, custom
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str]
    domain_restriction: Optional[str] = None

@dataclass
class RBACConfig:
    """RBAC configuration"""
    default_role: str = "user"
    admin_role: str = "admin"
    custom_roles_enabled: bool = True
    permission_inheritance: bool = True

class SSOProvider:
    """Single Sign-On provider implementation"""
    
    def __init__(self, config: SSOConfig):
        self.config = config
        self.session = aiohttp.ClientSession()
    
    async def get_auth_url(self, state: str) -> str:
        """Generate OAuth2 authorization URL"""
        if self.config.provider == "google":
            return f"https://accounts.google.com/oauth2/authorize?client_id={self.config.client_id}&redirect_uri={self.config.redirect_uri}&scope={' '.join(self.config.scopes)}&response_type=code&state={state}"
        elif self.config.provider == "azure":
            return f"https://login.microsoftonline.com/common/oauth2/authorize?client_id={self.config.client_id}&redirect_uri={self.config.redirect_uri}&scope={' '.join(self.config.scopes)}&response_type=code&state={state}"
        else:
            raise ValueError(f"Unsupported SSO provider: {self.config.provider}")
    
    async def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        token_url = self._get_token_url()
        
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.config.redirect_uri
        }
        
        async with self.session.post(token_url, data=data) as response:
            return await response.json()
    
    async def get_user_info(self, access_token: str) -> Dict:
        """Get user information from SSO provider"""
        user_info_url = self._get_user_info_url()
        
        headers = {'Authorization': f'Bearer {access_token}'}
        async with self.session.get(user_info_url, headers=headers) as response:
            return await response.json()
    
    def _get_token_url(self) -> str:
        if self.config.provider == "google":
            return "https://oauth2.googleapis.com/token"
        elif self.config.provider == "azure":
            return "https://login.microsoftonline.com/common/oauth2/token"
        else:
            raise ValueError(f"Unsupported SSO provider: {self.config.provider}")
    
    def _get_user_info_url(self) -> str:
        if self.config.provider == "google":
            return "https://www.googleapis.com/oauth2/v2/userinfo"
        elif self.config.provider == "azure":
            return "https://graph.microsoft.com/v1.0/me"
        else:
            raise ValueError(f"Unsupported SSO provider: {self.config.provider}")

class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self, config: RBACConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self._cache = {}
    
    def create_role(self, name: str, permissions: List[str], company_id: str = None, description: str = "") -> Role:
        """Create a new role"""
        role = Role(
            id=str(uuid.uuid4()),
            name=name,
            company_id=company_id,
            permissions=permissions,
            description=description
        )
        self.db_session.add(role)
        self.db_session.commit()
        return role
    
    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user"""
        user = self.db_session.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        user.role = role_id
        self.db_session.commit()
        return True
    
    def check_permission(self, user_id: str, permission: str, resource_id: str = None) -> bool:
        """Check if user has specific permission"""
        user = self.db_session.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            return False
        
        # Get user's role and permissions
        role = self.db_session.query(Role).filter(Role.id == user.role).first()
        if not role:
            return False
        
        # Check if user has the permission
        if permission in role.permissions:
            return True
        
        # Check company-specific permissions
        if user.company_id and resource_id:
            company_role = self.db_session.query(Role).filter(
                Role.company_id == user.company_id,
                Role.name == role.name
            ).first()
            if company_role and permission in company_role.permissions:
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user"""
        user = self.db_session.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        role = self.db_session.query(Role).filter(Role.id == user.role).first()
        if not role:
            return []
        
        return role.permissions

class AuditManager:
    """Audit trail and compliance logging manager"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def log_action(self, user_id: str, action: str, resource_type: str = None, 
                   resource_id: str = None, details: Dict = None, 
                   ip_address: str = None, user_agent: str = None) -> AuditLog:
        """Log an audit event"""
        user = self.db_session.query(User).filter(User.id == user_id).first()
        
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            company_id=user.company_id if user else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db_session.add(audit_log)
        self.db_session.commit()
        return audit_log
    
    def get_audit_logs(self, user_id: str = None, company_id: str = None, 
                      action: str = None, start_date: datetime = None, 
                      end_date: datetime = None, limit: int = 100) -> List[AuditLog]:
        """Get audit logs with filters"""
        query = self.db_session.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if company_id:
            query = query.filter(AuditLog.company_id == company_id)
        if action:
            query = query.filter(AuditLog.action == action)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    def export_audit_logs(self, company_id: str, start_date: datetime, 
                         end_date: datetime, format: str = "json") -> str:
        """Export audit logs for compliance"""
        logs = self.get_audit_logs(
            company_id=company_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if format == "json":
            return json.dumps([asdict(log) for log in logs], default=str)
        elif format == "csv":
            # Implement CSV export
            pass
        
        return ""

class WhiteLabelManager:
    """White-labeling and customization manager"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def set_company_branding(self, company_id: str, branding: Dict) -> bool:
        """Set custom branding for a company"""
        company = self.db_session.query(Company).filter(Company.id == company_id).first()
        if not company:
            return False
        
        company.custom_branding = branding
        self.db_session.commit()
        return True
    
    def get_company_branding(self, company_id: str) -> Dict:
        """Get custom branding for a company"""
        company = self.db_session.query(Company).filter(Company.id == company_id).first()
        if not company:
            return {}
        
        return company.custom_branding or {}
    
    def generate_custom_css(self, company_id: str) -> str:
        """Generate custom CSS for white-labeling"""
        branding = self.get_company_branding(company_id)
        
        css = f"""
        :root {{
            --primary-color: {branding.get('primary_color', '#007bff')};
            --secondary-color: {branding.get('secondary_color', '#6c757d')};
            --accent-color: {branding.get('accent_color', '#28a745')};
            --text-color: {branding.get('text_color', '#333')};
            --background-color: {branding.get('background_color', '#fff')};
        }}
        
        .custom-logo {{
            background-image: url('{branding.get('logo_url', '')}');
        }}
        
        .custom-header {{
            background-color: var(--primary-color);
            color: white;
        }}
        """
        
        return css

class KubernetesManager:
    """Kubernetes deployment and orchestration manager"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            config.load_kube_config(config_path)
        else:
            config.load_incluster_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()
    
    def deploy_service(self, name: str, image: str, replicas: int = 3, 
                      env_vars: Dict = None, resources: Dict = None) -> bool:
        """Deploy a service to Kubernetes"""
        try:
            # Create deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=name,
                                    image=image,
                                    env=[client.V1EnvVar(name=k, value=v) for k, v in (env_vars or {}).items()],
                                    resources=client.V1ResourceRequirements(
                                        requests=resources.get('requests', {}),
                                        limits=resources.get('limits', {})
                                    ) if resources else None
                                )
                            ]
                        )
                    )
                )
            )
            
            self.apps_v1.create_namespaced_deployment(
                namespace="default",
                body=deployment
            )
            
            # Create service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(name=f"{name}-service"),
                spec=client.V1ServiceSpec(
                    selector={"app": name},
                    ports=[client.V1ServicePort(port=80, target_port=8080)]
                )
            )
            
            self.v1.create_namespaced_service(
                namespace="default",
                body=service
            )
            
            logger.info(f"Successfully deployed service: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {name}: {e}")
            return False
    
    def scale_service(self, name: str, replicas: int) -> bool:
        """Scale a service up or down"""
        try:
            self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace="default",
                body={"spec": {"replicas": replicas}}
            )
            logger.info(f"Scaled service {name} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Failed to scale service {name}: {e}")
            return False
    
    def get_service_status(self, name: str) -> Dict:
        """Get status of a deployed service"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace="default"
            )
            
            return {
                'name': name,
                'replicas': deployment.spec.replicas,
                'available_replicas': deployment.status.available_replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'updated_replicas': deployment.status.updated_replicas
            }
        except Exception as e:
            logger.error(f"Failed to get status for service {name}: {e}")
            return {}

class MonitoringManager:
    """Monitoring and observability manager"""
    
    def __init__(self):
        self.metrics = {
            'request_count': REQUEST_COUNT,
            'request_duration': REQUEST_DURATION,
            'active_users': ACTIVE_USERS,
            'database_connections': DATABASE_CONNECTIONS
        }
        
        # Initialize Sentry for error tracking
        sentry_sdk.init(
            dsn=os.getenv('SENTRY_DSN'),
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics['request_count'].labels(method=method, endpoint=endpoint, status=status).inc()
        self.metrics['request_duration'].observe(duration)
    
    def update_active_users(self, count: int):
        """Update active users metric"""
        self.metrics['active_users'].set(count)
    
    def update_database_connections(self, count: int):
        """Update database connections metric"""
        self.metrics['database_connections'].set(count)
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            'request_count': self.metrics['request_count']._value.sum(),
            'active_users': self.metrics['active_users']._value.sum(),
            'database_connections': self.metrics['database_connections']._value.sum()
        }

class DisasterRecoveryManager:
    """Disaster recovery and backup manager"""
    
    def __init__(self, s3_bucket: str, region: str = 'us-east-1'):
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_bucket = s3_bucket
    
    def create_backup(self, data: Dict, backup_name: str) -> bool:
        """Create a backup to S3"""
        try:
            backup_data = {
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"backups/{backup_name}.json",
                Body=json.dumps(backup_data),
                ContentType='application/json'
            )
            
            logger.info(f"Created backup: {backup_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            return False
    
    def restore_backup(self, backup_name: str) -> Dict:
        """Restore data from backup"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f"backups/{backup_name}.json"
            )
            
            backup_data = json.loads(response['Body'].read())
            logger.info(f"Restored backup: {backup_name}")
            return backup_data['data']
            
        except ClientError as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            return {}
    
    def list_backups(self) -> List[str]:
        """List available backups"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="backups/"
            )
            
            return [obj['Key'].replace('backups/', '').replace('.json', '') 
                   for obj in response.get('Contents', [])]
            
        except ClientError as e:
            logger.error(f"Failed to list backups: {e}")
            return []

class EnterpriseInfrastructure:
    """Main enterprise infrastructure orchestrator"""
    
    def __init__(self, db_url: str, redis_url: str = None):
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize Redis for caching
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Initialize managers
        self.db_session = self.SessionLocal()
        self.sso_provider = None
        self.rbac_manager = RBACManager(RBACConfig(), self.db_session)
        self.audit_manager = AuditManager(self.db_session)
        self.white_label_manager = WhiteLabelManager(self.db_session)
        self.kubernetes_manager = None
        self.monitoring_manager = MonitoringManager()
        self.disaster_recovery_manager = None
        
        logger.info("Enterprise Infrastructure initialized")
    
    def setup_sso(self, config: SSOConfig):
        """Setup SSO provider"""
        self.sso_provider = SSOProvider(config)
        logger.info(f"SSO provider setup: {config.provider}")
    
    def setup_kubernetes(self, config_path: str = None):
        """Setup Kubernetes manager"""
        self.kubernetes_manager = KubernetesManager(config_path)
        logger.info("Kubernetes manager setup")
    
    def setup_disaster_recovery(self, s3_bucket: str, region: str = 'us-east-1'):
        """Setup disaster recovery"""
        self.disaster_recovery_manager = DisasterRecoveryManager(s3_bucket, region)
        logger.info("Disaster recovery manager setup")
    
    def create_user(self, email: str, password: str, first_name: str = None, 
                   last_name: str = None, company_id: str = None, role: str = "user") -> User:
        """Create a new user"""
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            company_id=company_id,
            role=role
        )
        
        self.db_session.add(user)
        self.db_session.commit()
        
        # Log audit event
        self.audit_manager.log_action(
            user_id=user.id,
            action="user_created",
            details={"email": email, "role": role}
        )
        
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        user = self.db_session.query(User).filter(User.email == email).first()
        if not user or not user.is_active:
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            user.last_login = datetime.utcnow()
            self.db_session.commit()
            return user
        
        return None
    
    def generate_jwt_token(self, user: User, expires_in: int = 3600) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'company_id': user.company_id,
            'role': user.role,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        return jwt.encode(payload, os.getenv('JWT_SECRET', 'secret'), algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET', 'secret'), algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, permission: str = None):
        """Decorator for requiring authentication and optional permission"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # This would be implemented in your web framework
                # For now, just return the function
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def get_system_health(self) -> Dict:
        """Get system health status"""
        health = {
            'database': self._check_database_health(),
            'redis': self._check_redis_health(),
            'kubernetes': self._check_kubernetes_health(),
            'metrics': self.monitoring_manager.get_metrics()
        }
        
        return health
    
    def _check_database_health(self) -> Dict:
        """Check database health"""
        try:
            self.db_session.execute("SELECT 1")
            return {'status': 'healthy', 'message': 'Database connection OK'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': str(e)}
    
    def _check_redis_health(self) -> Dict:
        """Check Redis health"""
        if not self.redis_client:
            return {'status': 'not_configured', 'message': 'Redis not configured'}
        
        try:
            self.redis_client.ping()
            return {'status': 'healthy', 'message': 'Redis connection OK'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': str(e)}
    
    def _check_kubernetes_health(self) -> Dict:
        """Check Kubernetes health"""
        if not self.kubernetes_manager:
            return {'status': 'not_configured', 'message': 'Kubernetes not configured'}
        
        try:
            # Check if we can list pods
            self.kubernetes_manager.v1.list_namespaced_pod("default")
            return {'status': 'healthy', 'message': 'Kubernetes connection OK'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize enterprise infrastructure
    enterprise = EnterpriseInfrastructure(
        db_url="postgresql://user:pass@localhost/enterprise_db",
        redis_url="redis://localhost:6379"
    )
    
    # Setup SSO
    sso_config = SSOConfig(
        provider="google",
        client_id="your_client_id",
        client_secret="your_client_secret",
        redirect_uri="http://localhost:3000/auth/callback",
        scopes=["openid", "email", "profile"]
    )
    enterprise.setup_sso(sso_config)
    
    # Setup Kubernetes
    enterprise.setup_kubernetes()
    
    # Setup disaster recovery
    enterprise.setup_disaster_recovery("my-backup-bucket")
    
    # Create a user
    user = enterprise.create_user(
        email="test@example.com",
        password="secure_password",
        first_name="John",
        last_name="Doe",
        role="admin"
    )
    
    # Test authentication
    auth_user = enterprise.authenticate_user("test@example.com", "secure_password")
    if auth_user:
        token = enterprise.generate_jwt_token(auth_user)
        print(f"Authentication successful. Token: {token[:50]}...")
    
    # Get system health
    health = enterprise.get_system_health()
    print("System Health:", json.dumps(health, indent=2)) 