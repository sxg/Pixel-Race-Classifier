from model import PixelModule
from data import PixelDataModule
from lightning.pytorch.cli import LightningCLI


def main():
    LightningCLI(PixelModule, PixelDataModule)


if __name__ == "__main__":
    main()
