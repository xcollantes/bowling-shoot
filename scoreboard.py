"""Score tracking and JSON persistence for bowling pin matches."""

import json
import logging as _logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import SCORES_FILE

logger = _logging.getLogger(__name__)


@dataclass
class Scoreboard:
    """Tracks wins for left and right players."""

    left_wins: int = 0
    right_wins: int = 0
    rounds: list[dict] = field(default_factory=list)

    def record_win(self, side: str) -> None:
        """
        Record a win for the given side.

        Args:
            side: Either "left" or "right".

        Raises:
            ValueError: If side is not "left" or "right".
        """
        if side not in ("left", "right"):
            raise ValueError(
                f"Invalid side '{side}'. Must be 'left' or 'right'."
            )

        if side == "left":
            self.left_wins += 1
        else:
            self.right_wins += 1

        self.rounds.append({
            "round": len(self.rounds) + 1,
            "winner": side,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def save(self, path: str = SCORES_FILE) -> None:
        """
        Save scoreboard to JSON file.

        Args:
            path: File path for the JSON file.
        """
        data = {
            "left_wins": self.left_wins,
            "right_wins": self.right_wins,
            "rounds": self.rounds,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str = SCORES_FILE) -> "Scoreboard":
        """
        Load scoreboard from JSON file.

        If the file doesn't exist or is corrupt, returns a
        fresh Scoreboard.

        Args:
            path: File path for the JSON file.

        Returns:
            Loaded or new Scoreboard instance.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(
                left_wins=data["left_wins"],
                right_wins=data["right_wins"],
                rounds=data.get("rounds", []),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            logger.warning(
                "Could not load scores from %s, starting fresh.",
                path,
            )
            return cls()

    def reset(self) -> None:
        """Reset all scores to zero."""
        self.left_wins = 0
        self.right_wins = 0
        self.rounds.clear()
