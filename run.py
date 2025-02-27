import hydra
from omegaconf import DictConfig
import RegularizingEmbeddings.lightning.train as train

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train.train(cfg)

if __name__ == "__main__":
    main()