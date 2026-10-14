"""Run on an internet-connected login node, never on GPU compute nodes."""

from reproduction.sync import main

if __name__ == "__main__":
    main()
