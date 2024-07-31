from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg

if __name__ == "__main__":
    deca = DECA(config=deca_cfg, device='cuda')