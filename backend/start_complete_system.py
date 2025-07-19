# Global warning suppression for clean startup
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*Kubernetes not available.*")
warnings.filterwarnings("ignore", message=".*stable-baselines3.*")
warnings.filterwarnings("ignore", message=".*wandb.*")

# Suppress specific library warnings
import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR) 