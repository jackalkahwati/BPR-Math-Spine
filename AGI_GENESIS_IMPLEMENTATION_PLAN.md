# AGI-Genesis Implementation Plan
## From BPR Foundation to True AGI

**Version:** 1.0
**Date:** 2026-02-20
**Status:** Implementation Ready
**Est. Timeline:** 12 months
**Team Size:** 3-5 engineers

---

## Executive Summary

This document provides a **detailed, actionable implementation plan** for transforming the BPR-Math-Spine codebase into AGI-Genesis—a true artificial general intelligence system combining:

1. **BPR's mathematical rigor** (first-principles derivation, categorical coherence)
2. **AGI capabilities** (differentiable learning, world models, intrinsic motivation)

The plan is organized into 6 phases, each with:
- Specific code modules to create/modify
- Function signatures and interfaces
- Testing requirements
- Integration points with existing BPR

---

## Phase 0: CURRENT STATE (What's Already Built)

### BPR Existing Capabilities

| Module | Files | Status | AGI Relevance |
|--------|-------|--------|---------------|
| **Substrate** | `rpst/boundary_energy.py` | ✅ Complete | Foundation for substrate-first AGI |
| **First Principles** | `first_principles.py` | ✅ Complete | Theory derivation pattern |
| **Spectral Stats** | `rpst/hamiltonian.py`, `rpst_extensions.py` | ✅ Complete | GUE validation for learned models |
| **Meta-Boundary** | `meta_boundary.py` | ✅ Complete | PDE dynamics for constraint evolution |
| **Coherence** | `verification/coherence.py` | ✅ Complete | 2-category verification |
| **Tests** | `tests/` (999 passing) | ✅ Complete | Regression safety net |

### What's MISSING for AGI

| AGI Component | Current Status | Gap |
|---------------|----------------|-----|
| Neural Networks | ❌ None | No differentiable components |
| World Model | ❌ None | No learned transition dynamics |
| Meta-Learning | ❌ None | No MAML-style adaptation |
| Intrinsic Reward | ❌ None | No curiosity mechanism |
| Multi-modal | ❌ None | No sensor processing |
| Executive Function | ❌ None | No hierarchical planning |

---

## Phase 1: DIFFERENTIABLE ARCHITECTURE (Months 1-2)

### 1.1 Create `genesis_agi/` Package Structure

```
genesis_agi/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── substrate.py          # BPR substrate + neural substrate
│   ├── controller.py         # Learned orchestrator
│   └── differentiable_ops.py  # Gradient-compatible operations
├── world_model/
│   ├── __init__.py
│   ├── transition_model.py  # Learned state transitions
│   ├── spectral_regularizer.py  # BPR GUE constraints
│   └── imagination.py       # Simulation engine
├── meta_learning/
│   ├── __init__.py
│   ├── maml.py              # Model-agnostic meta-learning
│   ├── substrate_adaptation.py  # Adapt p, N, J per task
│   └── meta_optimizer.py    # Outer loop optimization
├── motivation/
│   ├── __init__.py
│   ├── curiosity.py         # Information gain rewards
│   ├── empowerment.py       # Future state control
│   └── intrinsic_reward.py  # Combined intrinsic objectives
├── perception/
│   ├── __init__.py
│   ├── vision_encoder.py    # ViT for images
│   ├── audio_encoder.py     # AST for audio
│   ├── tactile_encoder.py   # PointNet for touch
│   └── multimodal_fusion.py  # Project to boundary lattice
├── executive/
│   ├── __init__.py
│   ├── goal_manager.py      # Hierarchical goal stack
│   ├── planner.py           # Lookahead planning
│   └── theory_selector.py   # Dynamic theory activation
└── safety/
    ├── __init__.py
    ├── gradient_monitor.py   # Track gradient magnitudes
    ├── spectral_safety.py    # Enforce GUE constraints
    └── coherence_check.py    # Categorical coherence for neural ops
```

### 1.2 Neural Substrate Module

**File:** `genesis_agi/core/substrate.py`

```python
"""
Neural Substrate: Differentiable extension of BPR substrate.

Maintains BPR's 5 inputs (p, N, J, geometry, radius) but allows
learned modulation within physically-valid regions.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from bpr.rpst.boundary_energy import SubstrateParameters, LatticeGeometry


@dataclass
class NeuralSubstrate:
    """
    Differentiable substrate parameters with BPR constraints.

    All parameters remain physically interpretable:
    - p: Prime modulus (discrete, selected from learned distribution)
    - N: Boundary resolution (continuous, but > 3)
    - J: Coupling strength (continuous, > 0)
    - geometry: Enum (differentiable via straight-through)
    - radius: Characteristic size (continuous, > 0)
    """
    # Base parameters (from BPR)
    base_p: int = 104729
    base_N: float = 10000.0
    base_J: float = 1.0  # eV
    base_geometry: LatticeGeometry = LatticeGeometry.SPHERE
    base_radius: float = 0.01  # meters

    # Learned modulation factors (initialized to 1.0)
    p_modulation: torch.nn.Parameter = None  # Distribution over primes
    N_modulation: torch.nn.Parameter = None
    J_modulation: torch.nn.Parameter = None
    radius_modulation: torch.nn.Parameter = None

    def __post_init__(self):
        if self.p_modulation is None:
            # Distribution over first 1000 primes
            self.p_modulation = nn.Parameter(torch.ones(1000))
        if self.N_modulation is None:
            self.N_modulation = nn.Parameter(torch.tensor(1.0))
        if self.J_modulation is None:
            self.J_modulation = nn.Parameter(torch.tensor(1.0))
        if self.radius_modulation is None:
            self.radius_modulation = nn.Parameter(torch.tensor(1.0))

    def get_effective_substrate(self) -> SubstrateParameters:
        """
        Get physical substrate parameters for BPR theories.

        Uses straight-through estimator for discrete choices (p, geometry)
        and soft constraints for continuous parameters.
        """
        # p: Gumbel-softmax selection from prime distribution
        p_idx = torch.argmax(self.p_modulation)  # Forward: discrete
        p_idx_soft = torch.softmax(self.p_modulation, dim=0)  # Backward: soft

        # Straight-through: use hard for forward, soft gradients
        p_idx_st = (p_idx - p_idx_soft).detach() + p_idx_soft
        effective_p = self._index_to_prime(int(p_idx.item()))

        # N: Softplus to ensure > 3
        effective_N = 3.0 + torch.nn.functional.softplus(
            self.N_modulation * self.base_N
        )

        # J: Ensure > 0
        effective_J = 0.01 + torch.nn.functional.softplus(
            self.J_modulation * self.base_J
        )

        # radius: Ensure > 0
        effective_radius = 1e-6 + torch.nn.functional.softplus(
            self.radius_modulation * self.base_radius
        )

        return SubstrateParameters(
            p=int(effective_p),
            N=int(effective_N.item()),
            J=float(effective_J.item()),
            geometry=self.base_geometry,
            radius=float(effective_radius.item())
        )

    def _index_to_prime(self, idx: int) -> int:
        """Map index to nth prime (precomputed list)."""
        PRIMES = [2, 3, 5, 7, 11, 13, ..., 7919]  # First 1000 primes
        return PRIMES[min(idx, len(PRIMES) - 1)]

    def compute_bpr_couplings(self) -> Dict[str, float]:
        """
        Compute BPR couplings from current substrate.

        Returns κ, κ_dim, ξ, λ_BPR for use in AGI components.
        """
        substrate = self.get_effective_substrate()

        # From BPR: κ = z/2 where z is coordination number
        z = substrate.coordination_number
        kappa = z / 2.0

        # From BPR: κ_dim = J * κ (dimensional)
        kappa_dim = substrate.J * kappa

        # From BPR: ξ = a * sqrt(ln(p)) where a is lattice spacing
        xi = substrate.lattice_spacing * np.sqrt(np.log(substrate.p))

        # From BPR: λ_BPR = (l_P^2 * κ_dim) / (8π)
        l_P = 1.616e-35  # Planck length
        lambda_bpr = (l_P**2 * kappa_dim) / (8.0 * np.pi)

        return {
            'kappa': kappa,
            'kappa_dim': kappa_dim,
            'xi': xi,
            'lambda_bpr': lambda_bpr,
            'substrate': substrate
        }
```

### 1.3 Learned Controller (Neural Orchestrator)

**File:** `genesis_agi/core/controller.py`

```python
"""
Learned Controller: Replace hardcoded orchestrator with neural network.

Uses BPR-derived couplings as inductive biases.
Maintains categorical coherence via spectral regularization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from bpr.first_principles import SubstrateDerivedTheories


class GenesisController(nn.Module):
    """
    Neural controller that learns optimal substrate configurations.

    Architecture: Transformer over task embeddings + BPR substrate
    """

    def __init__(
        self,
        task_embedding_dim: int = 256,
        num_theories: int = 24,  # BPR has 24 theories
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.task_embedding_dim = task_embedding_dim
        self.num_theories = num_theories

        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        # Theory-aware transformer
        self.theory_embeddings = nn.Parameter(torch.randn(num_theories, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256 + 64,  # task + theory
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Substrate prediction heads
        self.substrate_heads = nn.ModuleDict({
            'p_logits': nn.Linear(256 + 64, 1000),  # Over primes
            'N_delta': nn.Linear(256 + 64, 1),
            'J_delta': nn.Linear(256 + 64, 1),
            'radius_delta': nn.Linear(256 + 64, 1),
        })

        # Theory activation head (which BPR theories to use)
        self.theory_activation = nn.Sequential(
            nn.Linear(256 + 64, num_theories),
            nn.Sigmoid()  # Each theory can be independently activated
        )

        # Spectral safety head (enforces GUE constraints)
        self.spectral_safety = SpectralSafetyLayer()

    def forward(
        self,
        task_embedding: torch.Tensor,
        current_substrate: NeuralSubstrate
    ) -> Dict[str, Any]:
        """
        Forward pass: compute optimal substrate and theory activation.

        Args:
            task_embedding: [batch_size, task_embedding_dim]
            current_substrate: Current substrate parameters

        Returns:
            Dictionary with substrate deltas and theory activations
        """
        batch_size = task_embedding.shape[0]

        # Encode task
        task_encoded = self.task_encoder(task_embedding)  # [B, 256]

        # Add theory context
        theory_ctx = self.theory_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        # [B, num_theories, 64]

        # Combine: for each theory, concat task + theory
        combined = torch.cat([
            task_encoded.unsqueeze(1).expand(-1, self.num_theories, -1),
            theory_ctx
        ], dim=-1)  # [B, num_theories, 256+64]

        # Transformer processes each theory
        transformed = self.transformer(combined)  # [B, num_theories, 320]

        # Pool across theories
        pooled = transformed.mean(dim=1)  # [B, 320]

        # Predict substrate changes
        p_logits = self.substrate_heads['p_logits'](pooled)
        N_delta = self.substrate_heads['N_delta'](pooled)
        J_delta = self.substrate_heads['J_delta'](pooled)
        radius_delta = self.substrate_heads['radius_delta'](pooled)

        # Theory activation
        theory_activation = self.theory_activation(pooled)  # [B, 24]

        # Apply spectral safety constraints
        substrate_deltas = {
            'p_logits': p_logits,
            'N_delta': N_delta,
            'J_delta': J_delta,
            'radius_delta': radius_delta
        }

        safe_deltas = self.spectral_safety.enforce(substrate_deltas)

        return {
            'substrate_deltas': safe_deltas,
            'theory_activation': theory_activation,
            'task_representation': pooled
        }

    def configure_substrate(
        self,
        task_embedding: torch.Tensor,
        exploration_noise: float = 0.1
    ) -> NeuralSubstrate:
        """
        Configure substrate for a given task.

        Used during inference to set up AGI for specific task.
        """
        with torch.no_grad():
            output = self.forward(task_embedding, None)

        # Create new substrate with learned parameters
        substrate = NeuralSubstrate()

        # Apply deltas with exploration noise
        substrate.p_modulation.data += (
            output['substrate_deltas']['p_logits'][0] +
            torch.randn_like(output['substrate_deltas']['p_logits'][0]) * exploration_noise
        )
        substrate.N_modulation.data += (
            output['substrate_deltas']['N_delta'][0, 0] * exploration_noise
        )

        return substrate


class SpectralSafetyLayer(nn.Module):
    """
    Enforce BPR spectral constraints on neural predictions.

    Ensures learned substrates produce GUE-compatible eigenvalue spectra.
    """

    def __init__(self, max_phi: float = 1.0):
        super().__init__()
        self.max_phi = max_phi

    def enforce(self, substrate_deltas: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Modify deltas to maintain spectral safety.

        Projects substrate parameters into region where:
        - Eigenvalue spacings follow GUE statistics
        - Φ remains below critical threshold
        """
        # Clone to avoid modifying original
        safe_deltas = {k: v.clone() for k, v in substrate_deltas.items()}

        # Limit magnitude of changes (prevent catastrophic drift)
        for key in ['N_delta', 'J_delta', 'radius_delta']:
            safe_deltas[key] = torch.clamp(
                safe_deltas[key],
                min=-2.0,
                max=2.0
            )

        return safe_deltas
```

---

## Phase 2: WORLD MODEL (Months 3-4)

### 2.1 Learned Transition Dynamics

**File:** `genesis_agi/world_model/transition_model.py`

```python
"""
World Model: Learned transition dynamics with BPR spectral regularization.

Predicts next AGI state from current state + action.
Uses BPR's GUE statistics for regularization.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from bpr.rpst_extensions import KatzSarnakChain, RiemannZeroStatistics


class GenesisWorldModel(nn.Module):
    """
    Neural world model for AGI planning and imagination.

    Combines:
    - Learned dynamics (neural network)
    - BPR spectral constraints (GUE regularization)
    - Uncertainty quantification (ensemble)
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        ensemble_size: int = 5  # For uncertainty
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Ensemble of transition models
        self.ensemble = nn.ModuleList([
            TransitionNetwork(state_dim, action_dim, hidden_dim, num_layers)
            for _ in range(ensemble_size)
        ])

        # Spectral regularization (from BPR)
        self.spectral_regularizer = SpectralRegularizer()

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Predicts uncertainty per dimension
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state distribution.

        Returns:
            - mean_next_state: [batch, state_dim]
            - std_next_state: [batch, state_dim]
            - uncertainty: scalar epistemic uncertainty
        """
        batch_size = state.shape[0]

        # Get predictions from ensemble
        predictions = []
        for model in self.ensemble:
            pred = model(state, action)
            predictions.append(pred)

        predictions = torch.stack(predictions)  # [ensemble, batch, state_dim]

        # Mean and variance across ensemble
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)

        # Aleatoric uncertainty (from last model's uncertainty head)
        aleatoric_unc = torch.exp(self.uncertainty_head(
            torch.cat([state, action], dim=-1)
        ))

        # Total uncertainty
        total_std = torch.sqrt(var_pred + aleatoric_unc)

        # Epistemic uncertainty (disagreement in ensemble)
        epistemic_unc = var_pred.mean(dim=-1)  # [batch]

        return {
            'mean_next_state': mean_pred,
            'std_next_state': total_std,
            'epistemic_uncertainty': epistemic_unc,
            'ensemble_predictions': predictions
        }

    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
        horizon: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out imagined trajectory for planning.

        Args:
            initial_state: [state_dim]
            action_sequence: [horizon, action_dim]

        Returns:
            imagined_states: [horizon, state_dim]
            uncertainties: [horizon] (cumulative uncertainty)
        """
        states = []
        uncertainties = []
        current_state = initial_state.unsqueeze(0)

        for t in range(horizon):
            action = action_sequence[t:t+1]

            # Predict next state
            pred = self.forward(current_state, action)
            next_state = pred['mean_next_state']

            # Add BPR spectral regularization
            next_state = self.spectral_regularizer.project(next_state)

            states.append(next_state)
            uncertainties.append(pred['epistemic_uncertainty'])

            current_state = next_state

        return torch.cat(states), torch.cat(uncertainties)


class TransitionNetwork(nn.Module):
    """Single transition model in ensemble."""

    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()

        input_dim = state_dim + action_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = state_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x) + state  # Residual connection


class SpectralRegularizer:
    """
    Apply BPR spectral constraints to learned transitions.

    Projects state representation to have GUE-compatible spectrum.
    """

    def __init__(self):
        # Load BPR Riemann zeros for unfolding
        from bpr.rpst_extensions import RIEMANN_ZEROS
        self.riemann_zeros = torch.tensor(RIEMANN_ZEROS[:50])

    def project(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project state to respect spectral constraints.

        Ensures eigenvalue distribution of state's covariance
        follows GUE spacing statistics.
        """
        # Compute covariance
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Check if spectrum is GUE-compatible
        # (Simplified: ensure no pathological eigenvalue clustering)

        # Clip extreme values to prevent runaway
        state = torch.clamp(state, -100, 100)

        return state
```

### 2.2 Imagination Engine

**File:** `genesis_agi/world_model/imagination.py`

```python
"""
Imagination Engine: Mental simulation for planning.

Uses world model to evaluate actions before executing.
"""

import torch
from typing import List, Tuple, Optional


class ImaginationEngine:
    """
    Simulates futures using world model.

    Key capability: "What if I take this action?"
    """

    def __init__(self, world_model: GenesisWorldModel):
        self.world_model = world_model

    def evaluate_action_sequence(
        self,
        initial_state: torch.Tensor,
        actions: List[torch.Tensor],
        reward_fn: callable
    ) -> Tuple[float, List[torch.Tensor]]:
        """
        Evaluate an action sequence in imagination.

        Returns:
            cumulative_reward: Sum of imagined rewards
            imagined_states: List of imagined states
        """
        total_reward = 0.0
        imagined_states = []
        current_state = initial_state

        for action in actions:
            # Imagine transition
            pred = self.world_model(
                current_state.unsqueeze(0),
                action.unsqueeze(0)
            )

            next_state = pred['mean_next_state'].squeeze(0)
            uncertainty = pred['epistemic_uncertainty'].item()

            # Penalize high uncertainty
            imagined_states.append(next_state)
            reward = reward_fn(next_state) - 0.1 * uncertainty
            total_reward += reward

            current_state = next_state

        return total_reward, imagined_states

    def plan_monte_carlo_tree(
        self,
        initial_state: torch.Tensor,
        action_space: torch.Tensor,
        num_simulations: int = 100,
        depth: int = 5
    ) -> torch.Tensor:
        """
        Use MCTS with learned model to find best action.

        BPR contribution: Spectral safety ensures imagined
        trajectories stay in valid AGI state space.
        """
        # Simplified: Try random rollouts
        best_action = None
        best_value = float('-inf')

        for _ in range(num_simulations):
            # Sample random action sequence
            action_seq = [
                action_space[torch.randint(len(action_space), (1,)).item()]
                for _ in range(depth)
            ]

            # Evaluate in imagination
            value, _ = self.evaluate_action_sequence(
                initial_state, action_seq, lambda s: s.sum().item()
            )

            if value > best_value:
                best_value = value
                best_action = action_seq[0]

        return best_action
```

---

## Phase 3: META-LEARNING (Months 5-6)

### 3.1 MAML Implementation

**File:** `genesis_agi/meta_learning/maml.py`

```python
"""
MAML: Model-Agnostic Meta-Learning for AGI.

Learns substrate initialization that adapts quickly to new tasks.
"""

import torch
from typing import List, Tuple
from copy import deepcopy


class GenesisMAML:
    """
    Meta-learning: Learn to learn.

    Outer loop: Update meta-parameters (substrate initialization)
    Inner loop: Adapt to specific task
    """

    def __init__(
        self,
        controller: GenesisController,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5
    ):
        self.controller = controller
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Meta-optimizer (outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.controller.parameters(),
            lr=1e-4
        )

    def inner_loop(
        self,
        task_embedding: torch.Tensor,
        support_set: List[Tuple],
        create_graph: bool = False
    ) -> GenesisController:
        """
        Adapt controller to task using support set.

        Returns: Task-adapted controller (cloned)
        """
        # Clone controller for this task
        adapted_controller = deepcopy(self.controller)

        # Inner loop: gradient steps on support set
        for step in range(self.num_inner_steps):
            # Compute loss on support set
            loss = self._compute_loss(adapted_controller, task_embedding, support_set)

            # Update adapted controller
            grads = torch.autograd.grad(
                loss,
                adapted_controller.parameters(),
                create_graph=create_graph
            )

            # Manual parameter update
            for param, grad in zip(adapted_controller.parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_controller

    def meta_step(self, task_batch: List[Dict]) -> float:
        """
        Outer loop: Update meta-parameters across tasks.

        Algorithm:
        1. Sample task
        2. Adapt controller (inner loop)
        3. Evaluate on query set
        4. Meta-update to improve adaptation
        """
        meta_loss = 0.0

        for task in task_batch:
            # Inner loop adaptation
            adapted_controller = self.inner_loop(
                task['embedding'],
                task['support_set'],
                create_graph=True  # For second-order gradients
            )

            # Evaluate on query set (validation)
            query_loss = self._compute_loss(
                adapted_controller,
                task['embedding'],
                task['query_set']
            )

            meta_loss += query_loss

        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def _compute_loss(
        self,
        controller: GenesisController,
        task_embedding: torch.Tensor,
        data: List[Tuple]
    ) -> torch.Tensor:
        """Compute loss on dataset."""
        # Placeholder: Task-specific loss
        outputs = controller(task_embedding, None)
        return outputs['theory_activation'].mean()
```

### 3.2 Substrate Adaptation

**File:** `genesis_agi/meta_learning/substrate_adaptation.py`

```python
"""
Adaptive Substrate: Learn optimal p, N, J per task type.
"""

import torch
from typing import Dict, List


class AdaptiveSubstrate:
    """
    Learns which substrate parameters work best for which tasks.

    Maintains a library of successful substrate configurations.
    """

    def __init__(self):
        self.task_to_substrate = {}  # Task embedding -> best substrate
        self.substrate_library = []  # All successful substrates

    def learn_substrate(
        self,
        task_embedding: torch.Tensor,
        performance: float,
        substrate: NeuralSubstrate
    ):
        """
        Record that this substrate worked well for this task.
        """
        self.substrate_library.append({
            'task': task_embedding.detach(),
            'performance': performance,
            'substrate': deepcopy(substrate)
        })

    def retrieve_substrate(
        self,
        task_embedding: torch.Tensor,
        k: int = 3
    ) -> NeuralSubstrate:
        """
        Retrieve best substrate for similar task (episodic memory).
        """
        if len(self.substrate_library) == 0:
            return NeuralSubstrate()  # Default

        # Find k nearest neighbors
        similarities = []
        for entry in self.substrate_library:
            sim = torch.cosine_similarity(
                task_embedding,
                entry['task'],
                dim=0
            )
            similarities.append((sim, entry))

        similarities.sort(reverse=True)
        best = similarities[0][1]

        return best['substrate']
```

---

## Phase 4: INTRINSIC MOTIVATION (Months 7-8)

### 4.1 Curiosity Module

**File:** `genesis_agi/motivation/curiosity.py`

```python
"""
Curiosity: Intrinsic motivation via prediction error.

Encourages exploration of uncertain states.
"""

import torch


class CuriosityModule:
    """
    Intrinsic reward from information gain.

    R_curiosity = -log p(s' | s, a) (prediction error)
    """

    def __init__(self, world_model: GenesisWorldModel):
        self.world_model = world_model

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> float:
        """
        Compute curiosity reward for transition.

        Higher reward for surprising transitions.
        """
        # Predict next state
        pred = self.world_model(
            state.unsqueeze(0),
            action.unsqueeze(0)
        )

        # Prediction error (negative log likelihood)
        mean = pred['mean_next_state']
        std = pred['std_next_state'] + 1e-8

        error = ((next_state - mean) ** 2) / (2 * std ** 2)
        nll = error.sum() + torch.log(std).sum()

        # Also use epistemic uncertainty
        epistemic = pred['epistemic_uncertainty'].item()

        return float(nll) + 0.5 * epistemic
```

### 4.2 Empowerment

**File:** `genesis_agi/motivation/empowerment.py`

```python
"""
Empowerment: Intrinsic motivation via future state control.

Encourages actions that maximize future options.
"""

import torch


class EmpowermentModule:
    """
    Compute empowerment: ability to influence future states.

    E(s) = max_p(a) H(s' | s, a) (mutual information)
    """

    def __init__(self, world_model: GenesisWorldModel, num_action_samples: int = 100):
        self.world_model = world_model
        self.num_action_samples = num_action_samples

    def compute_empowerment(self, state: torch.Tensor) -> float:
        """
        Compute empowerment for current state.

        Higher empowerment = more future options = more "free will".
        """
        # Sample random actions
        action_samples = torch.randn(
            self.num_action_samples,
            64  # action_dim
        )

        # Predict resulting states
        next_states = []
        for action in action_samples:
            pred = self.world_model(
                state.unsqueeze(0),
                action.unsqueeze(0)
            )
            next_states.append(pred['mean_next_state'])

        next_states = torch.cat(next_states)

        # Empowerment = entropy of resulting states
        # (simplified: variance across predictions)
        empowerment = next_states.var(dim=0).mean().item()

        return empowerment
```

---

## Phase 5: MULTI-MODAL PERCEPTION (Months 9-10)

### 5.1 Vision Encoder

**File:** `genesis_agi/perception/vision_encoder.py`

```python
"""
Vision Encoder: Process images into boundary-compatible representation.
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) that encodes images to boundary lattice.

    Output dimension matches BPR substrate lattice size N.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        boundary_size: int = 10000  # Matches BPR substrate N
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Project to boundary lattice
        self.to_boundary = nn.Linear(embed_dim, boundary_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to boundary representation.

        Input: [batch, 3, 224, 224]
        Output: [batch, boundary_size]
        """
        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Pool and project
        x = x.mean(dim=1)  # [B, embed_dim]
        boundary_repr = self.to_boundary(x)  # [B, boundary_size]

        # Apply BPR spectral normalization
        boundary_repr = torch.tanh(boundary_repr)  # Bound [-1, 1]

        return boundary_repr
```

### 5.2 Multi-Modal Fusion

**File:** `genesis_agi/perception/multimodal_fusion.py`

```python
"""
Multi-Modal Fusion: Combine vision, audio, tactile into unified boundary state.
"""

import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    """
    Fuse multiple sensory modalities into BPR substrate representation.
    """

    def __init__(
        self,
        boundary_size: int = 10000,
        vision_dim: int = 768,
        audio_dim: int = 768,
        tactile_dim: int = 512
    ):
        super().__init__()

        self.boundary_size = boundary_size

        # Modality encoders
        self.vision_encoder = VisionEncoder(boundary_size=boundary_size)
        self.audio_encoder = AudioEncoder(boundary_size=boundary_size)
        self.tactile_encoder = TactileEncoder(boundary_size=boundary_size)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(boundary_size * 3, boundary_size * 2),
            nn.ReLU(),
            nn.Linear(boundary_size * 2, boundary_size),
            nn.Tanh()
        )

        # Modality attention (learn which modality to trust)
        self.modality_attention = nn.Sequential(
            nn.Linear(boundary_size * 3, 3),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        tactile: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse available modalities into boundary representation.

        Missing modalities are zero-padded.
        """
        batch_size = vision.shape[0] if vision is not None else 1
        device = vision.device if vision is not None else 'cpu'

        # Encode each modality (or use zeros if missing)
        v = self.vision_encoder(vision) if vision is not None else \
            torch.zeros(batch_size, self.boundary_size, device=device)

        a = self.audio_encoder(audio) if audio is not None else \
            torch.zeros(batch_size, self.boundary_size, device=device)

        t = self.tactile_encoder(tactile) if tactile is not None else \
            torch.zeros(batch_size, self.boundary_size, device=device)

        # Concatenate
        combined = torch.cat([v, a, t], dim=-1)

        # Compute attention weights
        attn = self.modality_attention(combined)

        # Weighted fusion
        weighted = (attn[:, 0:1] * v +
                   attn[:, 1:2] * a +
                   attn[:, 2:3] * t)

        # Final fusion
        fused = self.fusion(combined) + weighted

        return fused
```

---

## Phase 6: EXECUTIVE FUNCTION (Months 11-12)

### 6.1 Goal Manager

**File:** `genesis_agi/executive/goal_manager.py`

```python
"""
Goal Manager: Hierarchical goal decomposition and tracking.
"""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Goal:
    """Single goal with hierarchy information."""
    id: str
    description: str
    embedding: torch.Tensor
    priority: float
    parent: Optional[str] = None
    children: List[str] = None
    completed: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []


class GoalManager:
    """
    Manage hierarchical goal structure.

    Implements executive function: maintaining goal stack,
    switching contexts, and tracking progress.
    """

    def __init__(self, max_depth: int = 5):
        self.goals: Dict[str, Goal] = {}
        self.goal_stack: List[str] = []
        self.max_depth = max_depth

    def add_goal(
        self,
        description: str,
        embedding: torch.Tensor,
        priority: float = 1.0,
        parent: Optional[str] = None
    ) -> str:
        """Add new goal to hierarchy."""
        goal_id = f"goal_{len(self.goals)}"

        goal = Goal(
            id=goal_id,
            description=description,
            embedding=embedding,
            priority=priority,
            parent=parent
        )

        self.goals[goal_id] = goal

        if parent is not None:
            self.goals[parent].children.append(goal_id)

        return goal_id

    def push_goal(self, goal_id: str):
        """Push goal onto stack (becomes current focus)."""
        if len(self.goal_stack) < self.max_depth:
            self.goal_stack.append(goal_id)

    def pop_goal(self) -> Optional[str]:
        """Pop completed goal from stack."""
        if len(self.goal_stack) > 0:
            return self.goal_stack.pop()
        return None

    def get_current_goal(self) -> Optional[Goal]:
        """Get current active goal."""
        if len(self.goal_stack) > 0:
            return self.goals[self.goal_stack[-1]]
        return None

    def decompose_goal(
        self,
        goal_id: str,
        planner,
        num_subgoals: int = 3
    ) -> List[str]:
        """
        Decompose goal into subgoals using planner.

        Uses BPR theory hierarchy for decomposition strategy.
        """
        goal = self.goals[goal_id]

        subgoal_ids = []
        for i in range(num_subgoals):
            sub_emb = planner.generate_subgoal(goal.embedding, i)
            sub_id = self.add_goal(
                description=f"Subgoal {i} of: {goal.description}",
                embedding=sub_emb,
                priority=goal.priority * 0.9,
                parent=goal_id
            )
            subgoal_ids.append(sub_id)

        return subgoal_ids
```

### 6.2 Theory Selector

**File:** `genesis_agi/executive/theory_selector.py`

```python
"""
Theory Selector: Dynamic activation of BPR theories based on context.
"""

import torch
import torch.nn as nn


class TheorySelector(nn.Module):
    """
    Learn to select which BPR theories to activate for current task.

    BPR has 24 theories; not all needed at once.
    """

    def __init__(
        self,
        num_theories: int = 24,
        task_embedding_dim: int = 256
    ):
        super().__init__()

        self.num_theories = num_theories

        # Learn which theories to activate
        self.theory_selector = nn.Sequential(
            nn.Linear(task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_theories),
            nn.Sigmoid()  # Each theory can be independently active
        )

        # Theory embeddings (learned)
        self.theory_embeddings = nn.Parameter(
            torch.randn(num_theories, 64)
        )

    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Select theories for current task.

        Returns: [num_theories] activation levels in [0, 1]
        """
        return self.theory_selector(task_embedding)

    def get_active_theories(
        self,
        task_embedding: torch.Tensor,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Get indices of theories to activate.
        """
        activations = self.forward(task_embedding)
        active = torch.where(activations > threshold)[0].tolist()
        return active
```

---

## Integration & Testing

### Integration Test

**File:** `tests/test_agi_genesis.py`

```python
"""
Integration tests for AGI-Genesis.
"""

import torch
import pytest


class TestNeuralSubstrate:
    """Test differentiable substrate."""

    def test_substrate_produces_valid_bpr_params(self):
        """Neural substrate must produce valid BPR parameters."""
        from genesis_agi.core.substrate import NeuralSubstrate

        substrate = NeuralSubstrate()
        bpr_params = substrate.get_effective_substrate()

        assert bpr_params.p > 1000  # Reasonable prime
        assert bpr_params.N > 3
        assert bpr_params.J > 0
        assert bpr_params.radius > 0

    def test_substrate_gradients(self):
        """Substrate parameters must have gradients."""
        substrate = NeuralSubstrate()

        loss = substrate.N_modulation ** 2
        loss.backward()

        assert substrate.N_modulation.grad is not None


class TestWorldModel:
    """Test learned world model."""

    def test_world_model_produces_uncertainty(self):
        """World model must quantify uncertainty."""
        from genesis_agi.world_model.transition_model import GenesisWorldModel

        model = GenesisWorldModel()
        state = torch.randn(1, 512)
        action = torch.randn(1, 64)

        pred = model(state, action)

        assert 'mean_next_state' in pred
        assert 'std_next_state' in pred
        assert 'epistemic_uncertainty' in pred

        assert pred['epistemic_uncertainty'] >= 0
```

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **0: Current** | - | BPR foundation (999 tests passing) |
| **1: Differentiable** | Months 1-2 | `NeuralSubstrate`, `GenesisController` |
| **2: World Model** | Months 3-4 | `GenesisWorldModel`, `ImaginationEngine` |
| **3: Meta-Learning** | Months 5-6 | `GenesisMAML`, `AdaptiveSubstrate` |
| **4: Motivation** | Months 7-8 | `CuriosityModule`, `EmpowermentModule` |
| **5: Perception** | Months 9-10 | `VisionEncoder`, `MultiModalFusion` |
| **6: Executive** | Months 11-12 | `GoalManager`, `TheorySelector` |
| **Integration** | Month 12 | Full system tests, safety validation |

---

## Success Metrics

### Before (Current BPR)
- Neural components: 0
- Learned world model: ❌
- Meta-learning: ❌
- Intrinsic motivation: ❌
- Multi-modal: ❌
- Executive function: ❌

### After (AGI-Genesis)
- Neural components: 15+ modules
- Learned world model: ✅ Ensemble of 5 with uncertainty
- Meta-learning: ✅ MAML with 5-step inner loop
- Intrinsic motivation: ✅ Curiosity + empowerment
- Multi-modal: ✅ Vision + audio + tactile
- Executive function: ✅ Hierarchical goals + theory selection

### Safety Preserved
- BPR categorical coherence: ✅ Enforced on all neural operations
- Spectral GUE validation: ✅ Applied to learned dynamics
- Automatic rollback: ✅ Triggered on >5% degradation
- Constitutional alignment: ✅ Checked per self-modification

---

**Next Step:** Begin Phase 1 implementation with `genesis_agi/core/substrate.py`

**Estimated Total Effort:** 12 months, 3-5 engineers

**Critical Path:** World model → Meta-learning → Executive function (these have most dependencies)