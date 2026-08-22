# test_hydra.py

"""
In this file, is to test the hydra configuration for the project. 
The hydra configuration allows us to manage and override configurations easily, 
making it flexible for different environments and use cases.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the application with the given configuration.
    
    Args:
        cfg (DictConfig): The configuration object provided by Hydra.
    """
    # Print the configuration for debugging purposes
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    print("\n\n\n")

    print(cfg.vad)

if __name__ == "__main__":
    main()


# > python test_hydra.py vad.atten_lim_db=0