#!/usr/bin/env python3
"""
Deployment and infrastructure helpers
"""
import os
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
import yaml
from pathlib import Path

logger = logging.getLogger('portfolio_optimizer.deployment')

class DeploymentValidator:
    """Validate deployment readiness"""
    
    def __init__(self):
        self.checks = []
        self.logger = logging.getLogger('portfolio_optimizer.deployment.validator')
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment configuration"""
        results = {
            'passed': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        required_vars = [
            'SECRET_KEY', 'JWT_SECRET_KEY', 'DATABASE_URL',
            'FLASK_ENV', 'CORS_ORIGINS'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                results['errors'].append(f"Missing required environment variable: {var}")
                results['passed'] = False
            else:
                results['checks'].append(f"✓ {var} is configured")
                
                # Check for insecure values in production
                if os.getenv('FLASK_ENV') == 'production':
                    if var == 'SECRET_KEY' and len(value) < 32:
                        results['warnings'].append(f"SECRET_KEY should be at least 32 characters in production")
                    if var == 'CORS_ORIGINS' and value == '*':
                        results['errors'].append("CORS_ORIGINS should not be '*' in production")
                        results['passed'] = False
        
        # Check optional but recommended vars
        optional_vars = ['REDIS_URL', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'ENCRYPTION_KEY']
        for var in optional_vars:
            if os.getenv(var):
                results['checks'].append(f"✓ {var} is configured")
            else:
                results['warnings'].append(f"Optional variable {var} not configured")
        
        return results
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate that all dependencies are available"""
        results = {
            'passed': True,
            'checks': [],
            'errors': []
        }
        
        try:
            # Check Python packages
            import flask
            import sqlalchemy
            import redis
            import sklearn
            import tensorflow
            results['checks'].append("✓ All required Python packages available")
            
            # Check database connection
            from sqlalchemy import create_engine
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                try:
                    engine = create_engine(database_url)
                    with engine.connect() as conn:
                        conn.execute("SELECT 1")
                    results['checks'].append("✓ Database connection successful")
                except Exception as e:
                    results['errors'].append(f"Database connection failed: {e}")
                    results['passed'] = False
            
            # Check Redis connection
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                try:
                    r = redis.from_url(redis_url)
                    r.ping()
                    results['checks'].append("✓ Redis connection successful")
                except Exception as e:
                    results['errors'].append(f"Redis connection failed: {e}")
                    results['passed'] = False
            
        except ImportError as e:
            results['errors'].append(f"Missing required dependency: {e}")
            results['passed'] = False
        
        return results
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security configuration"""
        results = {
            'passed': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        # Check if running as root (security risk)
        if os.geteuid() == 0:
            results['warnings'].append("Running as root user - consider using a non-root user")
        else:
            results['checks'].append("✓ Not running as root user")
        
        # Check file permissions
        sensitive_files = ['.env', 'config.py']
        for filename in sensitive_files:
            filepath = Path(filename)
            if filepath.exists():
                stat = filepath.stat()
                if stat.st_mode & 0o077:  # Check if group/others have access
                    results['warnings'].append(f"File {filename} has overly permissive permissions")
                else:
                    results['checks'].append(f"✓ {filename} has secure permissions")
        
        # Check SSL/TLS configuration
        if os.getenv('FLASK_ENV') == 'production':
            if not os.getenv('SSL_CERT_PATH') and not os.getenv('SSL_KEY_PATH'):
                results['warnings'].append("SSL certificates not configured for production")
        
        return results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all deployment validations"""
        env_results = self.validate_environment()
        dep_results = self.validate_dependencies()
        sec_results = self.validate_security()
        
        overall_passed = all([
            env_results['passed'],
            dep_results['passed'],
            sec_results['passed']
        ])
        
        return {
            'overall_passed': overall_passed,
            'timestamp': datetime.utcnow().isoformat(),
            'environment': env_results,
            'dependencies': dep_results,
            'security': sec_results
        }


class DatabaseMigrationManager:
    """Manage database migrations and schema updates"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.logger = logging.getLogger('portfolio_optimizer.migrations')
    
    def create_backup(self) -> str:
        """Create database backup before migration"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = f"backup_{timestamp}.sql"
        
        try:
            # Extract database details from URL
            from sqlalchemy import create_engine
            engine = create_engine(self.database_url)
            db_name = engine.url.database
            
            # Use pg_dump for PostgreSQL
            cmd = [
                'pg_dump',
                self.database_url,
                '-f', backup_file,
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"Database backup created: {backup_file}")
                return backup_file
            else:
                raise Exception(f"Backup failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def run_migrations(self, backup: bool = True) -> Dict[str, Any]:
        """Run database migrations"""
        results = {
            'success': False,
            'backup_file': None,
            'migrations_run': [],
            'errors': []
        }
        
        try:
            # Create backup if requested
            if backup:
                results['backup_file'] = self.create_backup()
            
            # Initialize database (creates tables if they don't exist)
            from models.database import init_database
            init_database()
            
            results['success'] = True
            results['migrations_run'].append('init_database')
            self.logger.info("Database migrations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            results['errors'].append(str(e))
        
        return results


class KubernetesHelper:
    """Kubernetes deployment helpers"""
    
    def __init__(self):
        self.logger = logging.getLogger('portfolio_optimizer.k8s')
    
    def generate_deployment_config(self, 
                                 image_tag: str = 'latest',
                                 replicas: int = 2,
                                 resources: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate Kubernetes deployment configuration"""
        
        default_resources = {
            'requests': {'cpu': '100m', 'memory': '256Mi'},
            'limits': {'cpu': '500m', 'memory': '512Mi'}
        }
        
        resources = resources or default_resources
        
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'portfolio-optimizer-backend',
                'labels': {'app': 'portfolio-optimizer', 'component': 'backend'}
            },
            'spec': {
                'replicas': replicas,
                'selector': {'matchLabels': {'app': 'portfolio-optimizer', 'component': 'backend'}},
                'template': {
                    'metadata': {'labels': {'app': 'portfolio-optimizer', 'component': 'backend'}},
                    'spec': {
                        'containers': [{
                            'name': 'backend',
                            'image': f'portfolio-optimizer-backend:{image_tag}',
                            'ports': [{'containerPort': 5000, 'name': 'http'}],
                            'env': [
                                {'name': 'FLASK_ENV', 'value': 'production'},
                                {'name': 'DATABASE_URL', 'valueFrom': {
                                    'secretKeyRef': {'name': 'db-secret', 'key': 'url'}
                                }},
                                {'name': 'REDIS_URL', 'valueFrom': {
                                    'secretKeyRef': {'name': 'redis-secret', 'key': 'url'}
                                }},
                                {'name': 'SECRET_KEY', 'valueFrom': {
                                    'secretKeyRef': {'name': 'app-secret', 'key': 'secret-key'}
                                }},
                                {'name': 'JWT_SECRET_KEY', 'valueFrom': {
                                    'secretKeyRef': {'name': 'app-secret', 'key': 'jwt-secret'}
                                }}
                            ],
                            'resources': resources,
                            'livenessProbe': {
                                'httpGet': {'path': '/health/liveness', 'port': 5000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/health/readiness', 'port': 5000},
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3
                            }
                        }],
                        'imagePullSecrets': [{'name': 'docker-registry-secret'}]
                    }
                }
            }
        }
        
        return deployment
    
    def generate_service_config(self) -> Dict[str, Any]:
        """Generate Kubernetes service configuration"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'portfolio-optimizer-backend-service',
                'labels': {'app': 'portfolio-optimizer', 'component': 'backend'}
            },
            'spec': {
                'selector': {'app': 'portfolio-optimizer', 'component': 'backend'},
                'ports': [{
                    'port': 80,
                    'targetPort': 5000,
                    'name': 'http'
                }],
                'type': 'ClusterIP'
            }
        }
    
    def generate_hpa_config(self, min_replicas: int = 2, max_replicas: int = 10) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler configuration"""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'portfolio-optimizer-backend-hpa',
                'labels': {'app': 'portfolio-optimizer', 'component': 'backend'}
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'portfolio-optimizer-backend'
                },
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {'type': 'Utilization', 'averageUtilization': 70}
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {'type': 'Utilization', 'averageUtilization': 80}
                        }
                    }
                ]
            }
        }
    
    def save_configs(self, output_dir: str = 'k8s'):
        """Save all Kubernetes configurations to files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        configs = {
            'deployment.yaml': self.generate_deployment_config(),
            'service.yaml': self.generate_service_config(),
            'hpa.yaml': self.generate_hpa_config()
        }
        
        for filename, config in configs.items():
            filepath = Path(output_dir) / filename
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Generated {filepath}")


class LoadTestRunner:
    """Run load tests to verify deployment performance"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger('portfolio_optimizer.loadtest')
    
    def run_basic_load_test(self, 
                          concurrent_users: int = 10,
                          duration_seconds: int = 60) -> Dict[str, Any]:
        """Run basic load test"""
        results = {
            'start_time': datetime.utcnow().isoformat(),
            'concurrent_users': concurrent_users,
            'duration_seconds': duration_seconds,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'requests_per_second': 0.0,
            'errors': []
        }
        
        try:
            import threading
            import time
            import statistics
            
            response_times = []
            request_count = 0
            error_count = 0
            
            def worker():
                nonlocal request_count, error_count, response_times
                
                end_time = time.time() + duration_seconds
                while time.time() < end_time:
                    try:
                        start = time.time()
                        response = requests.get(f"{self.base_url}/health", timeout=10)
                        response_time = (time.time() - start) * 1000
                        
                        response_times.append(response_time)
                        request_count += 1
                        
                        if response.status_code != 200:
                            error_count += 1
                        
                        time.sleep(0.1)  # Small delay between requests
                        
                    except Exception as e:
                        error_count += 1
                        results['errors'].append(str(e))
            
            # Start worker threads
            threads = []
            for _ in range(concurrent_users):
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Calculate results
            results['total_requests'] = request_count
            results['successful_requests'] = request_count - error_count
            results['failed_requests'] = error_count
            
            if response_times:
                results['average_response_time'] = statistics.mean(response_times)
                results['requests_per_second'] = request_count / duration_seconds
            
            results['end_time'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            self.logger.error(f"Load test failed: {e}")
            results['errors'].append(str(e))
        
        return results


def create_deployment_package(version: str = None) -> str:
    """Create a deployment package with all necessary files"""
    from datetime import datetime
    import shutil
    import tarfile
    
    if not version:
        version = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    
    package_name = f"portfolio-optimizer-{version}"
    package_dir = Path(package_name)
    
    try:
        # Create package directory
        package_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        files_to_copy = [
            'app.py',
            'requirements.txt',
            'config.py',
            'models/',
            'routes/',
            'utils/',
            'schemas.py'
        ]
        
        for item in files_to_copy:
            source = Path(item)
            if source.exists():
                if source.is_dir():
                    shutil.copytree(source, package_dir / item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, package_dir / item)
        
        # Generate deployment configs
        k8s_helper = KubernetesHelper()
        k8s_helper.save_configs(str(package_dir / 'k8s'))
        
        # Create version info file
        version_info = {
            'version': version,
            'build_time': datetime.utcnow().isoformat(),
            'git_commit': get_git_commit(),
            'python_version': subprocess.check_output(['python', '--version']).decode().strip()
        }
        
        with open(package_dir / 'version.json', 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Create tarball
        tarball_name = f"{package_name}.tar.gz"
        with tarfile.open(tarball_name, 'w:gz') as tar:
            tar.add(package_dir, arcname=package_name)
        
        # Cleanup temporary directory
        shutil.rmtree(package_dir)
        
        logger.info(f"Deployment package created: {tarball_name}")
        return tarball_name
        
    except Exception as e:
        logger.error(f"Failed to create deployment package: {e}")
        # Cleanup on error
        if package_dir.exists():
            shutil.rmtree(package_dir)
        raise


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except:
        return 'unknown'