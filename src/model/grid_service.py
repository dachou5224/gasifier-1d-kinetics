import numpy as np
from dataclasses import dataclass
from .constants import PhysicalConstants

@dataclass
class MeshConfig:
    total_length: float
    n_cells: int
    ignition_zone_length: float = 0.4
    ignition_zone_res: float = 0.1
    min_grid_size: float = 0.1

class AdaptiveMeshGenerator:
    """
    Generates a 1D grid with variable resolution.
    Prioritizes resolution in the ignition zone.
    """
    def __init__(self, config: MeshConfig):
        self.cfg = config

    def generate(self):
        """
        Returns:
            dz_list (np.array): Array of cell lengths
            z_positions (np.array): Array of cell center positions
        """
        L = self.cfg.total_length
        N = self.cfg.n_cells
        dz_list = np.zeros(N)
        fixed_len = 0.0
        
        # Explicit Sizes derived from constants or config
        # Use config values, falling back to PhysicalConstants if not provided (though config has defaults)
        
        if N > 2:
            # 0. First Cell (Tuyere / Inlet)
            # Use defined ignition zone length first step or a specific large inlet cell?
            # Existing logic used FIRST_CELL_LENGTH (0.4)
            # Let's align with the config.
            dz_list[0] = self.cfg.ignition_zone_length
            fixed_len += self.cfg.ignition_zone_length
            
            # 1. Ignition/Flame Zone
            # Fill subsequent cells with fine resolution until we run out of cells or exceed length?
            # Existing logic: cells 1 to 20 get IGNITION_ZONE_DZ (0.1)
            fine_res = self.cfg.ignition_zone_res
            fine_end = min(20, N)
            
            for k in range(1, fine_end):
                dz_list[k] = fine_res
                fixed_len += fine_res
                
            # 2. Remaining Length Distributed
            rem_cells = N - fine_end
            if rem_cells > 0:
                rem_len = max(L - fixed_len, self.cfg.min_grid_size) # Safety
                dz_avg = rem_len / rem_cells
                for k in range(fine_end, N):
                    dz_list[k] = dz_avg
        else:
            # Fallback for small N
            dz_list[:] = L / N

        # Compute Z positions (Centers)
        z_positions = np.zeros(N)
        current_z = 0.0
        for k in range(N):
            z_positions[k] = current_z + dz_list[k]/2.0
            current_z += dz_list[k]
            
        return dz_list, z_positions
