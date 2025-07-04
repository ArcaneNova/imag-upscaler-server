# Celery Configuration for High Performance Real-ESRGAN Processing

# Broker settings - Force local Redis only
import os

# Force local Redis configuration only - ignore all external Redis settings
redis_host = "127.0.0.1"
redis_port = "6379"

# Always use local Redis - no fallbacks to external services
broker_url = f"redis://{redis_host}:{redis_port}/0"
result_backend = f"redis://{redis_host}:{redis_port}/0"

# Serialization
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

# Timezone
timezone = "UTC"
enable_utc = True

# Worker settings for optimal performance
worker_prefetch_multiplier = 1  # Process one task at a time to avoid memory issues
worker_max_tasks_per_child = 50  # Restart worker after 50 tasks to prevent memory leaks
worker_disable_rate_limits = False
worker_acks_late = True  # Acknowledge tasks after completion
worker_reject_on_worker_lost = True

# Task routing and priority
task_routes = {
    'upscale_image': {'queue': 'upscale'},
    'cleanup_old_jobs': {'queue': 'maintenance'},
    'health_check': {'queue': 'health'}
}

# Task execution settings
task_acks_late = True
task_reject_on_worker_lost = True
task_ignore_result = False
task_store_eager_result = True

# Result settings
result_expires = 3600  # Results expire after 1 hour
result_persistent = True

# Redis settings optimization
broker_transport_options = {
    'visibility_timeout': 3600,  # 1 hour
    'fanout_prefix': True,
    'fanout_patterns': True,
    'socket_keepalive': True,
    'socket_keepalive_options': {
        'TCP_KEEPIDLE': 1,
        'TCP_KEEPINTVL': 3,
        'TCP_KEEPCNT': 5,
    }
}

# Monitoring and logging
worker_send_task_events = True
task_send_sent_event = True
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'

# Security
worker_hijack_root_logger = False
worker_log_color = True

# Performance optimizations
broker_pool_limit = 10
broker_connection_timeout = 30
broker_connection_retry = True
broker_connection_max_retries = 100

# Task soft/hard time limits (in seconds)
task_soft_time_limit = 300  # 5 minutes soft limit
task_time_limit = 600       # 10 minutes hard limit

# Beat schedule for maintenance tasks
beat_schedule = {
    'cleanup-old-jobs': {
        'task': 'cleanup_old_jobs',
        'schedule': 3600.0,  # Every hour
    },
    'worker-health-check': {
        'task': 'health_check',
        'schedule': 300.0,   # Every 5 minutes
    },
}

# Queue definitions
task_default_queue = 'default'
task_default_exchange = 'default'
task_default_exchange_type = 'direct'
task_default_routing_key = 'default'

# Advanced settings for memory management
worker_max_memory_per_child = 1024000  # 1GB memory limit per worker
worker_disable_rate_limits = False
worker_enable_remote_control = True
worker_send_task_events = True
