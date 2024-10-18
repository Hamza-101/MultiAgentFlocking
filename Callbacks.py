import numpy as np
from tqdm import tqdm
from Parameters import *
from stable_baselines3.common.callbacks import BaseCallback

class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        
        # if(self.current_timestep > (SimulationVariables["LearningTimesteps"]/2)):
        if len(self.model.ep_info_buffer) >= 1000:
            recent_losses = [ep_info['loss'] for ep_info in self.model.ep_info_buffer[-1000:]]
            average_loss = np.mean(recent_losses)

            if average_loss < self.loss_threshold:
                print(f"Stopping training because average loss ({average_loss}) is below threshold.")
                return False  

        return True

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if self.pbar:
            # Update progress bar with the number of timesteps
            self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

