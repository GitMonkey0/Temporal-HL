from __future__ import annotations

from pathlib import Path


TABLE = """| Task | Model | Metrics |
|---|---|---|
| Notation translation | Static-HL baseline | static acc = 0.8311 |
| Notation translation | Temporal-HL baseline | static acc = 0.7902, motion acc = 0.6869, keyframe F1 = 0.4711 |
| Token-to-motion recon | Static-HL Transformer decoder | coord L1 = 0.0971 |
| Token-to-motion recon | Temporal-HL Transformer decoder | coord L1 = 0.0915 |
| Token-to-motion recon | Static-HL GRU decoder | coord L1 = 0.0584 |
| Token-to-motion recon | Temporal-HL GRU decoder | coord L1 = 0.0612 |
"""


def main() -> None:
    out = Path("paper_assets")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result_table.md").write_text(TABLE, encoding="utf-8")


if __name__ == "__main__":
    main()
