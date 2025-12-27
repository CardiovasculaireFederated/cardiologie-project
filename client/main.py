# client/main.py
import logging

from .manager_agent import ManagerAgent


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agent = ManagerAgent()
    agent.run()


if __name__ == "__main__":
    main()
