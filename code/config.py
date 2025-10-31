"""
Configuration file for Privacy-Enhanced Federated Learning
Easy-to-modify settings for privacy levels, algorithms, and training parameters
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class PrivacyConfig:
    """Privacy configuration settings"""
    
    # Privacy level options: 'conservative', 'moderate', 'liberal'
    privacy_level: str = 'moderate'
    
    # Differential Privacy settings
    enable_differential_privacy: bool = True
    default_epsilon: float = 1.0  # Privacy budget per round
    default_delta: float = 1e-5   # Approximate DP parameter
    
    # Gradient clipping settings
    gradient_clip_norm: float = 1.0
    adaptive_clipping: bool = True
    
    # Secure aggregation settings
    enable_secure_aggregation: bool = True
    byzantine_threshold: float = 0.25  # Can tolerate 25% malicious clients
    
    # Privacy budget allocation
    total_privacy_budget: float = 50.0  # Total ε across all rounds
    privacy_allocation_strategy: str = 'uniform'  # 'uniform', 'adaptive', 'conservative'

@dataclass 
class TrainingConfig:
    """Training configuration settings"""
    
    # Federated learning parameters
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.0005
    
    # Model parameters
    input_dimension: Optional[int] = None  # Will be auto-detected
    hidden_layers: List[int] = None  # [256, 128, 64]
    dropout_rate: float = 0.3
    
    # Algorithms to test
    algorithms: List[str] = None  # ['fedavg', 'fedprox', 'fedbn', 'ditto', 'scaffold']
    
    # Evaluation settings
    evaluation_frequency: int = 10  # Evaluate every N rounds
    save_frequency: int = 10  # Save every N rounds
    
    # Fairness analysis
    enable_fairness_analysis: bool = True
    fairness_metrics: List[str] = None  # ['equalized_odds', 'demographic_parity']

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]
        
        if self.algorithms is None:
            self.algorithms = ['fedavg', 'fedprox', 'fedbn', 'ditto', 'scaffold']
        
        if self.fairness_metrics is None:
            self.fairness_metrics = ['equalized_odds', 'demographic_parity']

@dataclass
class DataConfig:
    """Data configuration settings"""
    
    # Data paths
    processed_data_dir: Path = Path("processed_data_final_best")
    raw_data_path: Optional[Path] = None
    
    # Client configuration
    icu_types: List[str] = None  # ['MICU', 'SICU', 'CCU', 'CVICU', 'Neuro']
    clients_per_icu: int = 1  # Number of clients per ICU type
    
    # Data partitioning
    non_iid_partitioning: bool = True
    skew_factor: float = 0.1  # Degree of non-IID partitioning
    
    # Data quality checks
    enable_data_validation: bool = True
    min_samples_per_client: int = 100
    max_mortality_rate_difference: float = 0.2

    def __post_init__(self):
        if self.icu_types is None:
            self.icu_types = ['MICU', 'SICU', 'CCU', 'CVICU', 'Neuro']

@dataclass
class OutputConfig:
    """Output configuration settings"""
    
    # Output directories
    results_dir: Path = Path("privacy_fl_results")
    models_dir: Path = Path("privacy_fl_models") 
    logs_dir: Path = Path("privacy_fl_logs")
    
    # File naming
    experiment_name: str = "privacy_enhanced_fl"
    timestamp_results: bool = True
    
    # Save settings
    save_intermediate_results: bool = True
    save_final_models: bool = True
    save_training_history: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

class Config:
    """Master configuration class combining all settings"""
    
    def __init__(self, 
                 privacy_level: str = 'moderate',
                 num_rounds: int = 50,
                 algorithms: Optional[List[str]] = None):
        """
        Initialize complete configuration
        
        Args:
            privacy_level: 'conservative', 'moderate', or 'liberal'
            num_rounds: Number of federated training rounds
            algorithms: List of FL algorithms to test
        """
        self.privacy = PrivacyConfig(privacy_level=privacy_level)
        self.training = TrainingConfig(num_rounds=num_rounds, algorithms=algorithms)
        self.data = DataConfig()
        self.output = OutputConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        
        # Privacy budget validation
        if self.privacy.total_privacy_budget < self.training.num_rounds * self.privacy.default_epsilon:
            print(f"Warning: Total privacy budget ({self.privacy.total_privacy_budget}) may be insufficient "
                  f"for {self.training.num_rounds} rounds with ε={self.privacy.default_epsilon} per round")
        
        # Algorithm validation
        valid_algorithms = {'fedavg', 'fedprox', 'fedbn', 'ditto', 'scaffold'}
        for alg in self.training.algorithms:
            if alg not in valid_algorithms:
                raise ValueError(f"Invalid algorithm: {alg}. Must be one of {valid_algorithms}")
        
        # Privacy level validation
        valid_levels = {'conservative', 'moderate', 'liberal'}
        if self.privacy.privacy_level not in valid_levels:
            raise ValueError(f"Invalid privacy level: {self.privacy.privacy_level}. "
                           f"Must be one of {valid_levels}")
    
    def get_privacy_budget_per_round(self) -> float:
        """Calculate privacy budget per round"""
        if self.privacy.privacy_allocation_strategy == 'uniform':
            return self.privacy.total_privacy_budget / self.training.num_rounds
        elif self.privacy.privacy_allocation_strategy == 'conservative':
            return (self.privacy.total_privacy_budget / self.training.num_rounds) * 0.8
        elif self.privacy.privacy_allocation_strategy == 'adaptive':
            # Adaptive allocation based on training progress
            return self.privacy.total_privacy_budget / self.training.num_rounds
        else:
            return self.privacy.default_epsilon
    
    def get_client_configs(self) -> List[Dict]:
        """Generate client configurations"""
        client_configs = []
        
        for i, icu_type in enumerate(self.data.icu_types):
            for client_idx in range(self.data.clients_per_icu):
                client_id = f"client_{i}_{client_idx}" if self.data.clients_per_icu > 1 else f"client_{i}"
                client_configs.append({
                    'client_id': client_id,
                    'icu_type': icu_type,
                    'hospital_id': f'hospital_{i}',
                    'client_index': i
                })
        
        return client_configs
    
    def create_output_directories(self):
        """Create output directories"""
        self.output.results_dir.mkdir(parents=True, exist_ok=True)
        self.output.models_dir.mkdir(parents=True, exist_ok=True)
        self.output.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_name(self) -> str:
        """Get unique experiment name"""
        if self.output.timestamp_results:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.output.experiment_name}_{timestamp}"
        return self.output.experiment_name
    
    def save_config(self, output_dir: Path):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'privacy': asdict(self.privacy),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'output': asdict(self.output)
        }
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, config_file: Path) -> 'Config':
        """Load configuration from file"""
        import json
        from dataclasses import fromdict
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        privacy_config = fromdict(PrivacyConfig, config_dict['privacy'])
        training_config = fromdict(TrainingConfig, config_dict['training'])
        data_config = fromdict(DataConfig, config_dict['data'])
        output_config = fromdict(OutputConfig, config_dict['output'])
        
        # Create master config
        config = Config.__new__(Config)
        config.privacy = privacy_config
        config.training = training_config
        config.data = data_config
        config.output = output_config
        
        config._validate_config()
        return config

# Predefined configuration presets
def get_conservative_config() -> Config:
    """Get conservative privacy configuration (high privacy, lower utility)"""
    config = Config(privacy_level='conservative', num_rounds=50)
    config.privacy.default_epsilon = 0.5
    config.privacy.gradient_clip_norm = 0.8
    config.privacy.total_privacy_budget = 25.0
    return config

def get_moderate_config() -> Config:
    """Get moderate privacy configuration (balanced privacy and utility)"""
    config = Config(privacy_level='moderate', num_rounds=50)
    config.privacy.default_epsilon = 1.0
    config.privacy.gradient_clip_norm = 1.0
    config.privacy.total_privacy_budget = 50.0
    return config

def get_liberal_config() -> Config:
    """Get liberal privacy configuration (lower privacy, higher utility)"""
    config = Config(privacy_level='liberal', num_rounds=50)
    config.privacy.default_epsilon = 2.0
    config.privacy.gradient_clip_norm = 1.5
    config.privacy.total_privacy_budget = 100.0
    return config

def get_quick_test_config() -> Config:
    """Get quick test configuration (fewer rounds for testing)"""
    config = Config(privacy_level='moderate', num_rounds=10)
    config.training.algorithms = ['fedavg', 'fedprox']  # Test fewer algorithms
    return config

# Example usage
if __name__ == "__main__":
    # Create different configurations
    conservative = get_conservative_config()
    moderate = get_moderate_config() 
    liberal = get_liberal_config()
    
    print("Configuration Examples:")
    print(f"Conservative: ε={conservative.privacy.default_epsilon}, "
          f"budget={conservative.privacy.total_privacy_budget}")
    print(f"Moderate: ε={moderate.privacy.default_epsilon}, "
          f"budget={moderate.privacy.total_privacy_budget}")
    print(f"Liberal: ε={liberal.privacy.default_epsilon}, "
          f"budget={liberal.privacy.total_privacy_budget}")
    
    # Show client configurations
    print(f"\nClient configurations for moderate:")
    client_configs = moderate.get_client_configs()
    for config in client_configs[:3]:  # Show first 3
        print(f"  {config}")
    print(f"  ... ({len(client_configs)} total clients)")
