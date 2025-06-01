"""
Abstract base class for hyperparameter sweeps using wandb.

This module provides a minimal interface for implementing Bayesian hyperparameter
sweeps across different SAE trainers. Each trainer gets its own wandb project
for sweeps to keep things organized and focused.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime


class BaseSweepRunner(ABC):
    """
    Abstract base class for hyperparameter sweeps.
    
    Provides a standard interface for running Bayesian optimization sweeps
    using wandb. Each trainer type gets its own wandb project for organization.
    """
    
    def __init__(self, trainer_name: str, wandb_entity: str):
        """
        Initialize the sweep runner.
        
        Args:
            trainer_name: Name of the trainer (e.g. "vsae-iso", "vsae-gated")
            wandb_entity: wandb entity/username
        """
        self.trainer_name = trainer_name
        self.wandb_entity = wandb_entity
        self.sweep_project = f"{trainer_name}-sweeps"
        self.logger = logging.getLogger(__name__)
        
        # Create a group name for this sweep session
        self.sweep_group = f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    @abstractmethod
    def get_sweep_config(self) -> Dict[str, Any]:
        """
        Return the wandb sweep configuration dictionary.
        
        Should define the method (bayes), metric to optimize, and parameters
        to sweep over with their ranges/values.
        
        Returns:
            wandb sweep configuration dict
        """
        pass
    
    @abstractmethod
    def create_experiment_config(self, sweep_params: Dict[str, Any]):
        """
        Create an ExperimentConfig from wandb sweep parameters.
        
        Takes the parameters suggested by wandb and converts them into
        a full ExperimentConfig object for training.
        
        Args:
            sweep_params: Dictionary of parameters from wandb.config
            
        Returns:
            ExperimentConfig object ready for training
        """
        pass
    
    @abstractmethod
    def get_run_name(self, sweep_params: Dict[str, Any]) -> str:
        """
        Generate a descriptive run name from sweep parameters.
        
        Args:
            sweep_params: Dictionary of parameters from wandb.config
            
        Returns:
            Descriptive name for this specific run
        """
        pass
    
    def run_sweep(self) -> None:
        """
        Execute the wandb sweep.
        
        This method is the same for all trainers and handles the wandb
        sweep setup and execution. Each run gets a descriptive name and
        is organized in the trainer-specific project.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required for hyperparameter sweeps. Install with: pip install wandb")
        
        self.logger.info(f"Starting hyperparameter sweep for {self.trainer_name}")
        self.logger.info(f"Project: {self.sweep_project}")
        self.logger.info(f"Group: {self.sweep_group}")
        
        def train_with_sweep():
            """Inner function called by wandb agent for each sweep run."""
            # Initialize wandb first
            wandb.init(
                entity=self.wandb_entity,
                project=self.sweep_project,
                group=self.sweep_group,
                tags=["sweep", self.trainer_name, "bayesian-opt"]
            )
            
            # Now we can access wandb.config
            sweep_params = dict(wandb.config)
            run_name = self.get_run_name(sweep_params)
            
            # Update the run name now that we have the parameters
            wandb.run.name = run_name
            
            try:
                self.logger.info(f"Starting run: {run_name}")
                self.logger.info(f"Sweep parameters: {sweep_params}")
                
                # Clear GPU memory before starting
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create experiment config from sweep parameters
                config = self.create_experiment_config(sweep_params)
                
                # Import here to avoid circular imports
                from train_isovsae_1lgelu import ExperimentRunner
                
                # Run training with this configuration
                runner = ExperimentRunner(config)
                results = runner.run_training()
                
                # Log final results to wandb
                if results:
                    wandb.log(results)
                    self.logger.info(f"Run completed successfully: {run_name}")
                else:
                    self.logger.warning(f"No results returned for run: {run_name}")
                    
            except Exception as e:
                self.logger.error(f"Sweep run failed: {run_name} - {e}")
                # Log the error to wandb so we can track failed runs
                wandb.log({"error": str(e), "failed": True})
                raise
            finally:
                # Clean up memory after each run
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                wandb.finish()
        
        # Create and start the sweep
        sweep_config = self.get_sweep_config()
        self.logger.info(f"Sweep config: {sweep_config}")
        
        sweep_id = wandb.sweep(
            sweep_config, 
            entity=self.wandb_entity,
            project=self.sweep_project
        )
        
        self.logger.info(f"Created sweep with ID: {sweep_id}")
        self.logger.info(f"View sweep at: https://wandb.ai/{self.wandb_entity}/{self.sweep_project}/sweeps/{sweep_id}")
        self.logger.info(f"Starting wandb agent...")
        
        # Run the sweep agent
        wandb.agent(sweep_id, train_with_sweep)
        
        self.logger.info("Sweep completed")