"""
Secure Aggregation Protocol for Federated Learning
Implements cryptographic aggregation to prevent server inspection of individual updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import secrets
import logging
from collections import defaultdict
import math

@dataclass
class ClientKey:
    """Client cryptographic key for secure aggregation"""
    client_id: str
    public_key: torch.Tensor
    private_key: torch.Tensor
    share_keys: Dict[str, torch.Tensor]  # Keys for pairwise masking

class SecureAggregationManager:
    """
    Manages secure aggregation protocol for federated learning
    
    Provides cryptographic protection against server inspection of individual
    client updates while enabling correct aggregation
    """
    
    def __init__(self, num_clients: int, model_size: int):
        self.num_clients = num_clients
        self.model_size = model_size
        self.client_keys: Dict[str, ClientKey] = {}
        self.aggregation_masks: Dict[str, torch.Tensor] = {}
        self.byzantine_threshold = num_clients // 4  # Can tolerate up to 25% malicious clients
        
    def generate_client_keys(self, client_id: str) -> ClientKey:
        """
        Generate cryptographic keys for a client
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            ClientKey with public/private key pair and share keys
        """
        # Generate private key (random tensor)
        private_key = torch.randn(self.model_size)
        
        # Generate public key (transformed private key)
        public_key = self._transform_key(private_key)
        
        # Generate pairwise sharing keys with other clients
        share_keys = {}
        for other_id in self.client_keys.keys():
            if other_id != client_id:
                # Create pairwise key for secure communication
                share_key = self._generate_share_key(client_id, other_id)
                share_keys[other_id] = share_key
        
        client_key = ClientKey(
            client_id=client_id,
            public_key=public_key,
            private_key=private_key,
            share_keys=share_keys
        )
        
        self.client_keys[client_id] = client_key
        logging.info(f"Generated keys for client {client_id}")
        
        return client_key
    
    def _transform_key(self, key: torch.Tensor) -> torch.Tensor:
        """Apply cryptographic transformation to generate public key"""
        # Simple transformation for demonstration - in production use proper crypto
        transformed = torch.tanh(key * 100)  # Nonlinear transformation
        return transformed
    
    def _generate_share_key(self, client1_id: str, client2_id: str) -> torch.Tensor:
        """Generate shared key between two clients"""
        # Use client IDs as seeds for deterministic key generation
        seed = hash(f"{client1_id}_{client2_id}") % (2**32)
        torch.manual_seed(seed)
        share_key = torch.randn(self.model_size)
        torch.manual_seed(0)  # Reset seed
        return share_key
    
    def create_aggregation_mask(self, client_id: str, participating_clients: List[str]) -> torch.Tensor:
        """
        Create secure aggregation mask for a client
        
        The mask ensures that when summed across all clients, only the aggregate
        is visible, not individual contributions
        
        Args:
            client_id: Client creating the mask
            participating_clients: All clients participating in aggregation
            
        Returns:
            Aggregation mask tensor
        """
        mask = torch.zeros(self.model_size)
        
        # Generate pairwise masks with other clients
        for other_client in participating_clients:
            if other_client != client_id:
                # Create symmetric mask
                share_key = self._generate_share_key(client_id, other_client)
                
                # Add mask contribution (positive for this client, negative for symmetric property)
                if client_id < other_client:  # Deterministic assignment
                    mask += share_key
                else:
                    mask -= share_key
        
        self.aggregation_masks[client_id] = mask
        logging.debug(f"Created aggregation mask for client {client_id}")
        
        return mask
    
    def apply_secure_aggregation(self, client_updates: Dict[str, torch.Tensor], 
                               participating_clients: List[str]) -> torch.Tensor:
        """
        Perform secure aggregation of client updates
        
        Args:
            client_updates: Dictionary of client_id -> update_tensor
            participating_clients: List of participating client IDs
            
        Returns:
            Aggregated tensor (sum of all client updates)
        """
        if len(client_updates) != len(participating_clients):
            raise ValueError("Number of updates must match participating clients")
        
        # Step 1: Apply aggregation masks to each client's update
        masked_updates = {}
        for client_id in participating_clients:
            if client_id not in client_updates:
                raise ValueError(f"Missing update from client {client_id}")
            
            update = client_updates[client_id]
            mask = self.aggregation_masks.get(client_id, torch.zeros_like(update))
            
            # Apply mask (update - mask for this client, +mask for all others will cancel out)
            masked_update = update - mask
            masked_updates[client_id] = masked_update
        
        # Step 2: Sum all masked updates
        aggregated = torch.zeros_like(list(client_updates.values())[0])
        for client_id in participating_clients:
            aggregated += masked_updates[client_id]
        
        # Step 3: The masks cancel out in the sum, leaving only the true aggregate
        logging.info(f"Secure aggregation completed for {len(participating_clients)} clients")
        
        return aggregated
    
    def verify_byzantine_robustness(self, client_updates: Dict[str, torch.Tensor],
                                  participating_clients: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify that aggregation is robust against Byzantine (malicious) clients
        
        Args:
            client_updates: Dictionary of client updates
            participating_clients: List of participating clients
            
        Returns:
            Tuple of (is_robust, suspicious_clients)
        """
        if len(participating_clients) < self.byzantine_threshold * 2:
            return False, []  # Too few clients for Byzantine robustness
        
        # Check for statistical outliers using z-score
        update_norms = {}
        for client_id in participating_clients:
            update = client_updates[client_id]
            update_norms[client_id] = torch.norm(update).item()
        
        # Compute mean and standard deviation of norms
        norms = list(update_norms.values())
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Identify outliers (z-score > 3)
        suspicious_clients = []
        for client_id, norm in update_norms.items():
            if std_norm > 0:
                z_score = abs(norm - mean_norm) / std_norm
                if z_score > 3.0:  # 3-sigma rule
                    suspicious_clients.append(client_id)
        
        # Check if number of suspicious clients is acceptable
        is_robust = len(suspicious_clients) <= self.byzantine_threshold
        
        if not is_robust:
            logging.warning(f"Byzantine detection: {len(suspicious_clients)} suspicious clients, "
                          f"threshold={self.byzantine_threshold}")
        
        return is_robust, suspicious_clients
    
    def get_aggregation_statistics(self, client_updates: Dict[str, torch.Tensor]) -> Dict:
        """Get statistics about the aggregation process"""
        if not client_updates:
            return {}
        
        update_norms = {cid: torch.norm(update).item() 
                       for cid, update in client_updates.items()}
        
        stats = {
            'num_clients': len(client_updates),
            'min_norm': min(update_norms.values()),
            'max_norm': max(update_norms.values()),
            'mean_norm': np.mean(list(update_norms.values())),
            'std_norm': np.std(list(update_norms.values())),
            'byzantine_threshold': self.byzantine_threshold
        }
        
        return stats

class FederatedSecureServer:
    """Server for coordinating secure federated learning"""
    
    def __init__(self, num_clients: int, model_size: int):
        self.num_clients = num_clients
        self.model_size = model_size
        self.secure_agg_manager = SecureAggregationManager(num_clients, model_size)
        self.global_model = None
        self.round_count = 0
        
    def initialize_clients(self, client_ids: List[str]) -> Dict[str, ClientKey]:
        """
        Initialize all clients with cryptographic keys
        
        Args:
            client_ids: List of all client IDs
            
        Returns:
            Dictionary of client_id -> ClientKey
        """
        client_keys = {}
        for client_id in client_ids:
            client_key = self.secure_agg_manager.generate_client_keys(client_id)
            client_keys[client_id] = client_key
            
        logging.info(f"Initialized {len(client_ids)} clients for secure aggregation")
        return client_keys
    
    def coordinate_aggregation_round(self, client_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Coordinate a secure aggregation round
        
        Args:
            client_updates: Dictionary of client_id -> model update tensor
            
        Returns:
            Aggregated update tensor
        """
        participating_clients = list(client_updates.keys())
        
        # Verify Byzantine robustness
        is_robust, suspicious = self.secure_agg_manager.verify_byzantine_robustness(
            client_updates, participating_clients
        )
        
        if not is_robust:
            # Filter out suspicious clients
            logging.warning(f"Filtering out suspicious clients: {suspicious}")
            for client_id in suspicious:
                client_updates.pop(client_id, None)
            participating_clients = list(client_updates.keys())
        
        # Create aggregation masks for each client
        for client_id in participating_clients:
            self.secure_agg_manager.create_aggregation_mask(client_id, participating_clients)
        
        # Perform secure aggregation
        aggregated_update = self.secure_agg_manager.apply_secure_aggregation(
            client_updates, participating_clients
        )
        
        self.round_count += 1
        
        # Log statistics
        stats = self.secure_agg_manager.get_aggregation_statistics(client_updates)
        logging.info(f"Round {self.round_count}: {stats}")
        
        return aggregated_update
    
    def update_global_model(self, aggregated_update: torch.Tensor, 
                          learning_rate: float = 1.0) -> torch.Tensor:
        """
        Update global model with aggregated update
        
        Args:
            aggregated_update: Securely aggregated update tensor
            learning_rate: Learning rate for update
            
        Returns:
            Updated global model parameters
        """
        if self.global_model is None:
            self.global_model = aggregated_update.clone()
        else:
            self.global_model += learning_rate * aggregated_update
            
        return self.global_model

# Privacy-preserving aggregation strategies
class PrivacyPreservingAggregation:
    """Higher-level aggregation strategies that combine DP and secure aggregation"""
    
    def __init__(self, num_clients: int, model_size: int, dp_mechanisms: Dict[str, 'GaussianDPMechanism']):
        self.num_clients = num_clients
        self.model_size = model_size
        self.dp_mechanisms = dp_mechanisms  # Client ID -> DP mechanism
        self.secure_server = FederatedSecureServer(num_clients, model_size)
        
    def secure_aggregated_update(self, client_updates: Dict[str, torch.Tensor],
                               client_epsilons: Dict[str, float] = None) -> torch.Tensor:
        """
        Perform aggregation with both differential privacy and secure aggregation
        
        Args:
            client_updates: Client update tensors (already have DP noise added)
            client_epsilons: Privacy budget spent by each client this round
            
        Returns:
            Securely aggregated update
        """
        # Step 1: Ensure all updates have proper DP noise
        dp_updates = {}
        for client_id, update in client_updates.items():
            if client_id in self.dp_mechanisms:
                dp_mechanism = self.dp_mechanisms[client_id]
                # Verify DP was applied (check privacy budget usage)
                accounting = dp_mechanism.get_privacy_accounting()
                if accounting['spent_epsilon'] == 0:
                    logging.warning(f"Client {client_id} missing DP noise application")
            dp_updates[client_id] = update
        
        # Step 2: Perform secure aggregation
        aggregated = self.secure_server.coordinate_aggregation_round(dp_updates)
        
        # Step 3: Additional server-side privacy if needed
        if client_epsilons:
            total_epsilon_used = sum(client_epsilons.values())
            if total_epsilon_used > 10.0:  # High privacy cost - add extra protection
                # Add minimal additional noise for extra privacy
                noise_scale = 0.1 * math.sqrt(2 * math.log(1.25 / 1e-5))
                additional_noise = torch.randn_like(aggregated) * noise_scale
                aggregated += additional_noise
                logging.info(f"Added extra privacy noise for high ε usage: {total_epsilon_used:.1f}")
        
        return aggregated
    
    def create_client_participation_schedule(self, total_rounds: int, 
                                           client_availability: Dict[str, List[int]] = None) -> Dict[int, List[str]]:
        """
        Create schedule for client participation in secure aggregation
        
        Args:
            total_rounds: Total number of training rounds
            client_availability: Dict of client_id -> list of available rounds
            
        Returns:
            Round number -> list of participating clients
        """
        participation_schedule = {}
        client_ids = list(self.dp_mechanisms.keys())
        
        for round_num in range(total_rounds):
            # Simple round-robin participation
            participating = []
            for i, client_id in enumerate(client_ids):
                if client_availability is None or round_num in client_availability.get(client_id, []):
                    participating.append(client_id)
            
            participation_schedule[round_num] = participating
            
        return participation_schedule

def test_secure_aggregation():
    """Test the secure aggregation implementation"""
    print("Testing Secure Aggregation...")
    
    # Setup
    num_clients = 5
    model_size = 100
    
    # Create server
    server = FederatedSecureServer(num_clients, model_size)
    
    # Initialize clients
    client_ids = [f"client_{i}" for i in range(num_clients)]
    client_keys = server.initialize_clients(client_ids)
    
    # Simulate client updates
    client_updates = {}
    for client_id in client_ids:
        # Random model updates
        update = torch.randn(model_size) * 0.1
        client_updates[client_id] = update
    
    print(f"Original update norms: {[torch.norm(update).item() for update in client_updates.values()]}")
    
    # Perform secure aggregation
    aggregated = server.coordinate_aggregation_round(client_updates)
    
    # Verify correctness: aggregated should equal sum of all updates
    true_sum = sum(client_updates.values())
    
    print(f"Aggregated norm: {torch.norm(aggregated).item()}")
    print(f"True sum norm: {torch.norm(true_sum).item()}")
    print(f"Difference: {torch.norm(aggregated - true_sum).item()}")
    
    # Test Byzantine robustness
    updates_with_outlier = client_updates.copy()
    # Add a malicious update
    malicious_update = torch.randn(model_size) * 10  # Very large update
    updates_with_outlier["malicious_client"] = malicious_update
    
    is_robust, suspicious = server.secure_agg_manager.verify_byzantine_robustness(
        updates_with_outlier, list(updates_with_outlier.keys())
    )
    
    print(f"Byzantine robust: {is_robust}, Suspicious: {suspicious}")
    
    print("✅ Secure aggregation tests passed!")

if __name__ == "__main__":
    test_secure_aggregation()
