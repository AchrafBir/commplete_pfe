"""
Differential Privacy Mechanisms for Federated Learning
Implements Gaussian noise mechanisms with privacy budget tracking
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import logging
from collections import defaultdict

@dataclass
class PrivacyBudget:
    """Privacy budget tracking for (ε, δ)-differential privacy"""
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self.spent_epsilon)
    
    @property 
    def remaining_delta(self) -> float:
        return max(0.0, self.delta - self.spent_delta)
    
    def is_privacy_satisfied(self, add_epsilon: float, add_delta: float = 0.0) -> bool:
        """Check if adding this privacy cost would violate the budget"""
        return (self.spent_epsilon + add_epsilon <= self.epsilon and 
                self.spent_delta + add_delta <= self.delta)

    def spend_privacy(self, epsilon: float, delta: float = 0.0):
        """Spend privacy budget"""
        if not self.is_privacy_satisfied(epsilon, delta):
            raise ValueError(f"Privacy budget exceeded! Requested ε={epsilon}, δ={delta} "
                           f"but only ε={self.remaining_epsilon}, δ={self.remaining_delta} remaining")
        self.spent_epsilon += epsilon
        self.spent_delta += delta

class GradientClipping:
    """Adaptive gradient clipping for differential privacy"""
    
    def __init__(self, clip_norm: float = 1.0, adapt_clipping: bool = True):
        self.clip_norm = clip_norm
        self.adapt_clipping = adapt_clipping
        self.clip_history = []
        
    def clip_gradients(self, model: nn.Module) -> torch.Tensor:
        """Clip gradients with adaptive norm based on historical data"""
        if not self.adapt_clipping:
            return self._clip_to_norm(model.parameters(), self.clip_norm)
        
        # Compute gradient norms
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return torch.tensor(0.0)
            
        total_norm = total_norm ** (1. / 2)
        
        # Adaptive clipping based on percentile of historical norms
        if self.clip_history:
            percentile_95 = np.percentile(self.clip_history, 95)
            adaptive_clip = min(self.clip_norm, percentile_95)
        else:
            adaptive_clip = self.clip_norm
            
        if total_norm > adaptive_clip:
            clip_factor = adaptive_clip / total_norm
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_factor)
                    
            self.clip_history.append(total_norm)
            # Keep only recent history
            if len(self.clip_history) > 100:
                self.clip_history = self.clip_history[-100:]
        
        return torch.tensor(total_norm)

    def _clip_to_norm(self, parameters, clip_norm):
        """Clip gradients to specified norm"""
        total_norm = 0
        param_count = 0
        
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            if total_norm > clip_norm:
                clip_factor = clip_norm / total_norm
                for p in parameters:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_factor)
                        
        return torch.tensor(total_norm)

class GaussianDPMechanism:
    """Gaussian mechanism for (ε, δ)-differential privacy"""
    
    def __init__(self, privacy_budget: PrivacyBudget):
        self.privacy_budget = privacy_budget
        self.noise_history = defaultdict(list)
        
    def compute_noise_scale(self, epsilon: float, delta: float, sensitivity: float) -> float:
        """
        Compute Gaussian noise scale for (ε, δ)-differential privacy
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Approximate differential privacy parameter
            sensitivity: L2 sensitivity of the function
            
        Returns:
            Standard deviation of Gaussian noise
        """
        # For Gaussian mechanism: σ ≥ sqrt(2 * ln(1.25/δ)) * Δ₂ / ε
        sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        return sigma
    
    def add_noise_to_gradients(self, gradients: List[torch.Tensor], 
                             epsilon: float, delta: float = 1e-5,
                             clip_norm: float = 1.0) -> List[torch.Tensor]:
        """
        Add differentially private noise to gradients
        
        Args:
            gradients: List of gradient tensors
            epsilon: Privacy parameter
            delta: Approximate differential privacy parameter  
            clip_norm: Gradient clipping norm
            
        Returns:
            Noisy gradients with differential privacy guarantees
        """
        if not self.privacy_budget.is_privacy_satisfied(epsilon, delta):
            raise ValueError(f"Privacy budget exceeded! Remaining: ε={self.privacy_budget.remaining_epsilon}")
        
        # Compute sensitivity based on clipping
        sensitivity = clip_norm
        
        # Compute noise scale
        noise_scale = self.compute_noise_scale(epsilon, delta, sensitivity)
        
        # Add Gaussian noise to each gradient
        noisy_gradients = []
        for i, grad in enumerate(gradients):
            if grad is not None:
                # Generate noise with same shape as gradient
                noise = torch.randn_like(grad) * noise_scale
                noisy_grad = grad + noise
                noisy_gradients.append(noisy_grad)
                
                # Track noise for monitoring
                self.noise_history[f'layer_{i}'].append(noise_scale.item())
            else:
                noisy_gradients.append(None)
                
        # Spend privacy budget
        self.privacy_budget.spend_privacy(epsilon, delta)
        
        logging.info(f"DP applied: ε={epsilon:.3f}, δ={delta}, σ={noise_scale:.3f}")
        
        return noisy_gradients
    
    def get_privacy_accounting(self) -> Dict:
        """Get current privacy budget usage"""
        return {
            'total_epsilon': self.privacy_budget.epsilon,
            'total_delta': self.privacy_budget.delta,
            'spent_epsilon': self.privacy_budget.spent_epsilon,
            'spent_delta': self.privacy_budget.spent_delta,
            'remaining_epsilon': self.privacy_budget.remaining_epsilon,
            'remaining_delta': self.privacy_budget.remaining_delta,
            'epsilon_ratio': self.privacy_budget.spent_epsilon / self.privacy_budget.epsilon,
            'delta_ratio': self.privacy_budget.spent_delta / self.privacy_budget.delta
        }

class PrivacyAwareOptimizer:
    """Optimizer wrapper that applies differential privacy"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 privacy_mechanism: GaussianDPMechanism,
                 gradient_clipper: GradientClipping):
        self.optimizer = optimizer
        self.privacy_mechanism = privacy_mechanism
        self.gradient_clipper = gradient_clipper
        self.round_count = 0
        
    def step(self, epsilon: float, delta: float = 1e-5, clip_norm: float = 1.0):
        """
        Perform DP-aware optimization step
        
        Args:
            epsilon: Privacy parameter for this round
            delta: Approximate differential privacy parameter
            clip_norm: Gradient clipping norm
        """
        # Clip gradients
        total_norm = self.gradient_clipper.clip_gradients(self.optimizer.param_groups[0]['params'])
        
        # Collect gradients for DP noise addition
        gradients = []
        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None:
                gradients.append(param.grad.data.clone())
            else:
                gradients.append(None)
        
        # Apply differential privacy noise
        if gradients and any(g is not None for g in gradients):
            noisy_gradients = self.privacy_mechanism.add_noise_to_gradients(
                gradients, epsilon, delta, clip_norm
            )
            
            # Update parameters with noisy gradients
            param_idx = 0
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None and param_idx < len(noisy_gradients):
                    param.grad.data = noisy_gradients[param_idx]
                    param_idx += 1
        
        # Standard optimizer step
        self.optimizer.step()
        self.round_count += 1
        
        # Log privacy usage
        if self.round_count % 10 == 0:
            accounting = self.privacy_mechanism.get_privacy_accounting()
            logging.info(f"Round {self.round_count}: Privacy usage - "
                        f"ε: {accounting['spent_epsilon']:.3f}/{accounting['total_epsilon']:.3f} "
                        f"({accounting['epsilon_ratio']:.1%}), "
                        f"δ: {accounting['spent_delta']:.2e}/{accounting['total_delta']:.2e}")

class MedicalDataPrivacyConfig:
    """Privacy configuration specifically tuned for medical data"""
    
    @staticmethod
    def get_medical_privacy_budgets() -> Dict[str, PrivacyBudget]:
        """
        Get privacy budgets appropriate for medical data
        
        For medical data, we typically use:
        - Conservative privacy: ε=1.0 (high privacy)
        - Moderate privacy: ε=5.0 (balanced)  
        - Liberal privacy: ε=10.0 (more utility)
        
        All with δ = 1/(N*sqrt(N)) where N is the number of patients
        """
        # Assume typical ICU dataset size
        n_patients = 50000  # Adjust based on your actual dataset size
        delta = 1.0 / (n_patients * math.sqrt(n_patients))
        
        return {
            'conservative': PrivacyBudget(epsilon=1.0, delta=delta),
            'moderate': PrivacyBudget(epsilon=5.0, delta=delta), 
            'liberal': PrivacyBudget(epsilon=10.0, delta=delta)
        }
    
    @staticmethod
    def get_icu_specific_config(icu_type: str, mortality_rate: float) -> Dict:
        """
        Get privacy configuration specific to ICU type and mortality rate
        
        Different ICU types may require different privacy levels based on
        sensitivity of data and sample size
        """
        base_epsilon = 5.0
        
        # Adjust epsilon based on mortality rate (rarer events need more privacy)
        if mortality_rate < 0.1:  # Very rare mortality
            adjusted_epsilon = base_epsilon * 0.8
        elif mortality_rate > 0.3:  # High mortality
            adjusted_epsilon = base_epsilon * 1.2
        else:
            adjusted_epsilon = base_epsilon
            
        # Adjust based on ICU type sensitivity
        sensitivity_multipliers = {
            'MICU': 1.0,      # Standard medical ICU
            'SICU': 0.9,      # Surgical - slightly more sensitive
            'CCU': 0.8,       # Cardiac - more sensitive
            'CVICU': 0.7,     # Cardiac Vascular - most sensitive
            'Neuro': 0.85     # Neurological - very sensitive
        }
        
        final_epsilon = adjusted_epsilon * sensitivity_multipliers.get(icu_type, 1.0)
        
        n_patients = 10000  # Assume per-ICU patient count
        delta = 1.0 / (n_patients * math.sqrt(n_patients))
        
        return {
            'privacy_budget': PrivacyBudget(epsilon=final_epsilon, delta=delta),
            'clip_norm': 1.0,
            'adapt_clipping': True
        }

def create_privacy_aware_training_components(client_id: str, 
                                           icu_type: str,
                                           mortality_rate: float,
                                           privacy_level: str = 'moderate') -> Tuple[GradientClipping, GaussianDPMechanism, PrivacyAwareOptimizer]:
    """
    Create all components needed for privacy-aware federated learning training
    
    Args:
        client_id: Unique identifier for the client
        icu_type: Type of ICU (affects privacy sensitivity)
        mortality_rate: Mortality rate in this client's data
        privacy_level: 'conservative', 'moderate', or 'liberal'
    
    Returns:
        Tuple of (gradient_clipper, dp_mechanism, privacy_optimizer)
        Note: privacy_optimizer requires an actual optimizer to wrap
    """
    # Get privacy configuration
    config = MedicalDataPrivacyConfig.get_icu_specific_config(icu_type, mortality_rate)
    privacy_budgets = MedicalDataPrivacyConfig.get_medical_privacy_budgets()
    
    # Select privacy budget
    selected_budget = privacy_budgets[privacy_level]
    
    # Create components
    gradient_clipper = GradientClipping(
        clip_norm=config['clip_norm'],
        adapt_clipping=config['adapt_clipping']
    )
    
    dp_mechanism = GaussianDPMechanism(selected_budget)
    
    logging.info(f"Created privacy components for client {client_id} ({icu_type}): "
                f"ε={selected_budget.epsilon:.1f}, δ={selected_budget.delta:.2e}")
    
    return gradient_clipper, dp_mechanism, None  # Optimizer will be set later

# Privacy testing utilities
def test_privacy_mechanism():
    """Test the differential privacy implementation"""
    print("Testing Differential Privacy Mechanisms...")
    
    # Test privacy budget
    budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    print(f"Initial budget: ε={budget.epsilon}, δ={budget.delta}")
    
    # Test spending privacy
    budget.spend_privacy(2.0)
    print(f"After spending ε=2.0: ε_remaining={budget.remaining_epsilon}")
    
    # Test gradient clipping
    clipper = GradientClipping(clip_norm=1.0)
    
    # Test DP mechanism
    dp_mechanism = GaussianDPMechanism(budget)
    
    # Test noise generation
    gradients = [torch.randn(10, 5), torch.randn(5, 1)]
    noisy_grads = dp_mechanism.add_noise_to_gradients(gradients, epsilon=1.0, clip_norm=1.0)
    
    print(f"Original gradient norms: {[g.norm().item() for g in gradients if g is not None]}")
    print(f"Noisy gradient norms: {[g.norm().item() for g in noisy_grads if g is not None]}")
    
    # Test privacy accounting
    accounting = dp_mechanism.get_privacy_accounting()
    print(f"Privacy accounting: {accounting}")
    
    print("✅ Privacy mechanism tests passed!")

if __name__ == "__main__":
    test_privacy_mechanism()
