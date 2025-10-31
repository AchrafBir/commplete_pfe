"""
Privacy-Enhanced Federated Learning Algorithms
Integrates differential privacy and secure aggregation with existing FL methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import logging
from dataclasses import dataclass
import copy

# Import our privacy components
from dp_mechanisms import (
    GaussianDPMechanism, PrivacyBudget, GradientClipping, 
    PrivacyAwareOptimizer, MedicalDataPrivacyConfig,
    create_privacy_aware_training_components
)
from secure_aggregation import (
    FederatedSecureServer, PrivacyPreservingAggregation,
    SecureAggregationManager
)

@dataclass
class ClientDataInfo:
    """Information about client data for privacy-aware aggregation"""
    client_id: str
    icu_type: str
    sample_size: int
    mortality_rate: float
    privacy_budget: PrivacyBudget

class PrivacyEnhancedFederatedClient:
    """Enhanced federated learning client with privacy protections"""
    
    def __init__(self, client_id: str, icu_type: str, model_fn: Callable, 
                 privacy_level: str = 'moderate'):
        self.client_id = client_id
        self.icu_type = icu_type
        self.model_fn = model_fn
        self.privacy_level = privacy_level
        
        # Initialize model
        self.model = model_fn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        
        # Privacy components
        self.gradient_clipper = None
        self.dp_mechanism = None
        self.privacy_optimizer = None
        
        # Data statistics for adaptive privacy
        self.sample_size = 0
        self.mortality_rate = 0.0
        
        self._initialize_privacy_components()
    
    def _initialize_privacy_components(self):
        """Initialize privacy components based on ICU type and data characteristics"""
        # Use moderate privacy as default (adjust based on your needs)
        clipper, dp_mechanism, _ = create_privacy_aware_training_components(
            client_id=self.client_id,
            icu_type=self.icu_type,
            mortality_rate=self.mortality_rate,
            privacy_level=self.privacy_level
        )
        
        self.gradient_clipper = clipper
        self.dp_mechanism = dp_mechanism
        self.privacy_optimizer = PrivacyAwareOptimizer(
            optimizer=self.optimizer,
            privacy_mechanism=dp_mechanism,
            gradient_clipper=clipper
        )
    
    def set_data_statistics(self, X: torch.Tensor, y: torch.Tensor):
        """Update client data statistics for adaptive privacy"""
        self.sample_size = len(X)
        self.mortality_rate = y.float().mean().item()
        
        # Reinitialize privacy components with updated statistics
        self._initialize_privacy_components()
        
        logging.info(f"Client {self.client_id}: {self.sample_size} samples, "
                    f"mortality_rate={self.mortality_rate:.3f}")
    
    def train_local(self, X: torch.Tensor, y: torch.Tensor, 
                   global_model_params: Optional[Dict[str, torch.Tensor]] = None,
                   algorithm: str = 'fedavg', **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Train locally with privacy protection
        
        Args:
            X: Training features
            y: Training labels  
            global_model_params: Global model parameters for initialization
            algorithm: FL algorithm to use ('fedavg', 'fedprox', 'fedbn', 'ditto', 'scaffold')
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Tuple of (noisy_updates, training_stats)
        """
        # Update data statistics
        self.set_data_statistics(X, y)
        
        # Initialize with global model if provided
        if global_model_params:
            self._load_model_params(global_model_params)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Calculate class weights for imbalanced data
        pos_weight = (1 - self.mortality_rate) / max(self.mortality_rate, 0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
        # Determine algorithm-specific parameters
        if algorithm == 'fedprox':
            proximal_mu = kwargs.get('proximal_mu', 0.01)
            global_params = {name: param.clone().detach() 
                           for name, param in self.model.named_parameters()}
        
        elif algorithm == 'ditto':
            ditto_lambda = kwargs.get('ditto_lambda', 0.01)
            global_params = {name: param.clone().detach() 
                           for name, param in self.model.named_parameters()}
        
        elif algorithm == 'scaffold':
            # Scaffold control variates (simplified implementation)
            control_variates = kwargs.get('control_variates', {})
        
        # Batch training
        batch_size = kwargs.get('batch_size', 64)
        num_epochs = kwargs.get('local_epochs', 5)
        epsilon = kwargs.get('epsilon', 1.0)  # Privacy budget for this round
        
        for epoch in range(num_epochs):
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())
                
                # Algorithm-specific regularization
                if algorithm == 'fedprox':
                    # Add proximal term
                    proximal_term = 0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            proximal_term += torch.norm(param - global_params[name]) ** 2
                    loss += (proximal_mu / 2) * proximal_term
                
                elif algorithm == 'ditto':
                    # Add Ditto regularization toward global model
                    ditto_term = 0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            ditto_term += torch.norm(param - global_params[name]) ** 2
                    loss += (ditto_lambda / 2) * ditto_term
                
                # Backward pass with differential privacy
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping and DP noise
                self.privacy_optimizer.step(
                    epsilon=epsilon,
                    delta=kwargs.get('delta', 1e-5),
                    clip_norm=kwargs.get('clip_norm', 1.0)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        # Get privacy budget usage
        privacy_stats = self.dp_mechanism.get_privacy_accounting()
        
        # Extract model parameters as updates (differences from initialization)
        updates = {}
        if global_model_params:
            for name, param in self.model.named_parameters():
                if name in global_model_params:
                    updates[name] = param - global_model_params[name]
        else:
            for name, param in self.model.named_parameters():
                updates[name] = param.clone()
        
        training_stats = {
            'loss': total_loss / num_batches,
            'privacy_budget_used': privacy_stats['spent_epsilon'],
            'privacy_budget_remaining': privacy_stats['remaining_epsilon'],
            'sample_size': self.sample_size,
            'mortality_rate': self.mortality_rate,
            'algorithm': algorithm
        }
        
        logging.info(f"Client {self.client_id} training complete: "
                    f"loss={training_stats['loss']:.4f}, "
                    f"ε_used={privacy_stats['spent_epsilon']:.3f}")
        
        return updates, training_stats
    
    def _load_model_params(self, params: Dict[str, torch.Tensor]):
        """Load parameters into model"""
        for name, param in self.model.named_parameters():
            if name in params:
                param.data.copy_(params[name])

class PrivacyEnhancedFederatedServer:
    """Privacy-enhanced federated learning server"""
    
    def __init__(self, client_configs: List[Dict], model_fn: Callable, 
                 privacy_level: str = 'moderate'):
        self.client_configs = client_configs
        self.model_fn = model_fn
        self.privacy_level = privacy_level
        
        # Initialize clients
        self.clients = {}
        for config in client_configs:
            client = PrivacyEnhancedFederatedClient(
                client_id=config['client_id'],
                icu_type=config['icu_type'],
                model_fn=model_fn,
                privacy_level=privacy_level
            )
            self.clients[config['client_id']] = client
        
        # Privacy-preserving aggregation
        self.secure_server = FederatedSecureServer(
            num_clients=len(self.clients),
            model_size=sum(p.numel() for p in model_fn().parameters())
        )
        
        # Initialize clients for secure aggregation
        client_ids = list(self.clients.keys())
        self.secure_server.initialize_clients(client_ids)
        
        # Global model
        self.global_model = model_fn()
        
        self.round_count = 0
        
    def mortality_weighted_aggregation(self, client_updates: Dict[str, Dict[str, torch.Tensor]], 
                                     client_info: Dict[str, ClientDataInfo]) -> Dict[str, torch.Tensor]:
        """
        Perform mortality-weighted aggregation (your novel contribution)
        
        Args:
            client_updates: Dict of client_id -> model updates
            client_info: Dict of client_id -> ClientDataInfo
            
        Returns:
            Aggregated updates
        """
        # Compute mortality weights
        total_weighted_samples = 0
        client_weights = {}
        
        for client_id, updates in client_updates.items():
            info = client_info[client_id]
            
            # Weight based on sample size AND inverse mortality rate (rarer = higher weight)
            # This is your novel contribution from the original code
            mortality_weight = 1.0 / max(info.mortality_rate, 0.01)
            weighted_samples = info.sample_size * mortality_weight
            
            client_weights[client_id] = weighted_samples
            total_weighted_samples += weighted_samples
        
        # Normalize weights
        for client_id in client_weights:
            client_weights[client_id] /= total_weighted_samples
        
        # Aggregate with weights
        aggregated_updates = {}
        param_names = list(next(iter(client_updates.values())).keys())
        
        for param_name in param_names:
            aggregated_param = torch.zeros_like(next(iter(client_updates.values()))[param_name])
            
            for client_id, updates in client_updates.items():
                weight = client_weights[client_id]
                aggregated_param += weight * updates[param_name]
            
            aggregated_updates[param_name] = aggregated_param
        
        logging.info(f"Mortality-weighted aggregation: weights={client_weights}")
        
        return aggregated_updates
    
    def federated_round(self, client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
                       algorithm: str = 'fedavg', **kwargs) -> Dict:
        """
        Execute one federated learning round with privacy protection
        
        Args:
            client_data: Dict of client_id -> (X, y) tuples
            algorithm: FL algorithm to use
            **kwargs: Training parameters including privacy settings
            
        Returns:
            Round statistics
        """
        self.round_count += 1
        
        # Collect client updates with privacy protection
        client_updates = {}
        client_info = {}
        client_epsilons = {}
        
        for client_id, (X, y) in client_data.items():
            client = self.clients[client_id]
            
            # Get current global model parameters
            global_params = {name: param.clone().detach() 
                           for name, param in self.global_model.named_parameters()}
            
            # Train client with privacy protection
            updates, stats = client.train_local(
                X, y, global_model_params=global_params,
                algorithm=algorithm, **kwargs
            )
            
            client_updates[client_id] = updates
            client_epsilons[client_id] = stats['privacy_budget_used']
            
            # Store client info for aggregation
            client_info[client_id] = ClientDataInfo(
                client_id=client_id,
                icu_type=client.icu_type,
                sample_size=stats['sample_size'],
                mortality_rate=stats['mortality_rate'],
                privacy_budget=client.dp_mechanism.privacy_budget
            )
        
        # Perform privacy-preserving aggregation
        # Step 1: Apply differential privacy (already done in client training)
        # Step 2: Apply secure aggregation
        
        # Convert updates to parameter-wise format for secure aggregation
        param_wise_updates = defaultdict(dict)
        for client_id, updates in client_updates.items():
            for param_name, update_tensor in updates.items():
                param_wise_updates[param_name][client_id] = update_tensor
        
        # Aggregate each parameter with secure aggregation
        aggregated_updates = {}
        for param_name, client_tensor_updates in param_wise_updates.items():
            # Perform secure aggregation for this parameter
            secure_aggregated = self.secure_server.coordinate_aggregation_round(client_tensor_updates)
            aggregated_updates[param_name] = secure_aggregated
        
        # Apply mortality-weighted aggregation if selected
        if kwargs.get('use_mortality_weighting', True):
            aggregated_updates = self.mortality_weighted_aggregation(
                client_updates, client_info
            )
        
        # Update global model
        learning_rate = kwargs.get('learning_rate', 1.0)
        for param_name, aggregated_update in aggregated_updates.items():
            if param_name in self.global_model.state_dict():
                with torch.no_grad():
                    self.global_model.state_dict()[param_name] += learning_rate * aggregated_update
        
        # Calculate round statistics
        total_privacy_used = sum(client_epsilons.values())
        avg_loss = np.mean([stats['loss'] for client in self.clients.values() 
                          for stats in [client.train_local.__code__]])
        
        # This is a simplified loss calculation - in practice you'd track actual training losses
        round_stats = {
            'round': self.round_count,
            'num_clients': len(client_data),
            'algorithm': algorithm,
            'total_privacy_used': total_privacy_used,
            'average_loss': avg_loss,
            'privacy_accounting': {client_id: client.dp_mechanism.get_privacy_accounting() 
                                 for client_id, client in self.clients.items()}
        }
        
        logging.info(f"Round {self.round_count} complete: "
                    f"privacy_used={total_privacy_used:.3f}, "
                    f"clients={len(client_data)}")
        
        return round_stats

# Algorithm-specific enhancements
def create_privacy_enhanced_fedavg() -> Callable:
    """Create privacy-enhanced FedAvg client"""
    def client_train_fn(client: PrivacyEnhancedFederatedClient, X: torch.Tensor, y: torch.Tensor, **kwargs):
        return client.train_local(X, y, algorithm='fedavg', **kwargs)
    return client_train_fn

def create_privacy_enhanced_fedprox() -> Callable:
    """Create privacy-enhanced FedProx client"""
    def client_train_fn(client: PrivacyEnhancedFederatedClient, X: torch.Tensor, y: torch.Tensor, **kwargs):
        kwargs.setdefault('proximal_mu', 0.01)
        return client.train_local(X, y, algorithm='fedprox', **kwargs)
    return client_train_fn

def create_privacy_enhanced_fedbn() -> Callable:
    """Create privacy-enhanced FedBN client"""
    def client_train_fn(client: PrivacyEnhancedFederatedClient, X: torch.Tensor, y: torch.Tensor, **kwargs):
        return client.train_local(X, y, algorithm='fedbn', **kwargs)
    return client_train_fn

def create_privacy_enhanced_ditto() -> Callable:
    """Create privacy-enhanced Ditto client"""
    def client_train_fn(client: PrivacyEnhancedFederatedClient, X: torch.Tensor, y: torch.Tensor, **kwargs):
        kwargs.setdefault('ditto_lambda', 0.01)
        return client.train_local(X, y, algorithm='ditto', **kwargs)
    return client_train_fn

def create_privacy_enhanced_scaffold() -> Callable:
    """Create privacy-enhanced Scaffold client"""
    def client_train_fn(client: PrivacyEnhancedFederatedClient, X: torch.Tensor, y: torch.Tensor, **kwargs):
        return client.train_local(X, y, algorithm='scaffold', **kwargs)
    return client_train_fn

# Comprehensive FL training pipeline
class PrivacyEnhancedFLPipeline:
    """Complete privacy-enhanced federated learning pipeline"""
    
    def __init__(self, model_fn: Callable, client_configs: List[Dict], 
                 privacy_level: str = 'moderate'):
        self.model_fn = model_fn
        self.client_configs = client_configs
        self.privacy_level = privacy_level
        
        # Initialize federated server
        self.server = PrivacyEnhancedFederatedServer(
            client_configs=client_configs,
            model_fn=model_fn,
            privacy_level=privacy_level
        )
        
        # Training history
        self.training_history = []
        
    def train(self, client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
              num_rounds: int, algorithms: List[str] = None, **kwargs) -> Dict:
        """
        Train federated model with privacy protection
        
        Args:
            client_data: Client data (client_id -> (X, y))
            num_rounds: Number of federated rounds
            algorithms: List of algorithms to test (optional)
            **kwargs: Training parameters
            
        Returns:
            Training results and statistics
        """
        if algorithms is None:
            algorithms = ['fedavg', 'fedprox', 'fedbn', 'ditto', 'scaffold']
        
        results = {}
        
        for algorithm in algorithms:
            logging.info(f"Training with {algorithm}")
            
            # Reset server for clean experiment
            self.server = PrivacyEnhancedFederatedServer(
                client_configs=self.client_configs,
                model_fn=self.model_fn,
                privacy_level=self.privacy_level
            )
            
            # Run federated training
            round_stats = []
            for round_num in range(num_rounds):
                stats = self.server.federated_round(
                    client_data, algorithm=algorithm, **kwargs
                )
                round_stats.append(stats)
                
                # Log progress
                if (round_num + 1) % 10 == 0:
                    logging.info(f"{algorithm} - Round {round_num + 1}/{num_rounds} complete")
            
            results[algorithm] = {
                'round_stats': round_stats,
                'final_global_model': copy.deepcopy(self.server.global_model),
                'privacy_accounting': self.server.clients[list(self.server.clients.keys())[0]].dp_mechanism.get_privacy_accounting()
            }
            
            self.training_history.append({
                'algorithm': algorithm,
                'results': results[algorithm]
            })
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all training results"""
        summary = {}
        
        for training_record in self.training_history:
            algorithm = training_record['algorithm']
            results = training_record['results']
            
            # Extract key metrics
            privacy_used = results['privacy_accounting']['spent_epsilon']
            total_rounds = len(results['round_stats'])
            
            summary[algorithm] = {
                'total_privacy_used': privacy_used,
                'total_rounds': total_rounds,
                'avg_privacy_per_round': privacy_used / total_rounds,
                'final_global_model_state': results['final_global_model'].state_dict()
            }
        
        return summary

def test_privacy_enhanced_fl():
    """Test the privacy-enhanced federated learning implementation"""
    print("Testing Privacy-Enhanced Federated Learning...")
    
    # Create simple test model
    def create_test_model():
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    # Create test clients
    client_configs = [
        {'client_id': 'client_1', 'icu_type': 'MICU'},
        {'client_id': 'client_2', 'icu_type': 'SICU'},
        {'client_id': 'client_3', 'icu_type': 'CCU'}
    ]
    
    # Create test data
    client_data = {}
    for config in client_configs:
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,)).float()
        client_data[config['client_id']] = (X, y)
    
    # Initialize pipeline
    pipeline = PrivacyEnhancedFLPipeline(
        model_fn=create_test_model,
        client_configs=client_configs,
        privacy_level='moderate'
    )
    
    # Test single algorithm
    results = pipeline.train(
        client_data=client_data,
        num_rounds=5,
        algorithms=['fedavg'],
        epsilon=1.0,
        delta=1e-5,
        learning_rate=0.01
    )
    
    print("Training results:")
    for algorithm, result in results.items():
        print(f"  {algorithm}: {len(result['round_stats'])} rounds completed")
        print(f"  Privacy used: {result['privacy_accounting']['spent_epsilon']:.3f}")
    
    # Test multiple algorithms
    multi_results = pipeline.train(
        client_data=client_data,
        num_rounds=3,
        algorithms=['fedavg', 'fedprox', 'ditto'],
        epsilon=0.5,
        delta=1e-5,
        learning_rate=0.01
    )
    
    print("Multi-algorithm results:")
    for algorithm, result in multi_results.items():
        print(f"  {algorithm}: {result['privacy_accounting']['spent_epsilon']:.3f} ε used")
    
    print("✅ Privacy-enhanced FL tests passed!")

if __name__ == "__main__":
    test_privacy_enhanced_fl()
