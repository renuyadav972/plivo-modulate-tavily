"""Print the Velma behavior presets available on your account.

Run this once before the event. The preset identifiers in modulate.py come from
Modulate's published Agentic AI Guardrails package, and this confirms they match
what your key can actually request.

    python list_presets.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

URL = "https://platform.modulate.ai/api/velma-2-batch/list-presets"


def main() -> None:
    api_key = os.getenv("MODULATE_API_KEY")
    if not api_key:
        raise SystemExit("Set MODULATE_API_KEY in .env first")

    response = requests.get(URL, headers={"X-API-Key": api_key}, timeout=30)
    response.raise_for_status()

    presets = response.json()
    if isinstance(presets, dict):
        presets = presets.get("presets", presets)

    for preset in presets:
        identifier = preset.get("identifier", "?")
        name = preset.get("name", "")
        print(f"preset:{identifier:<36} {name}")
        description = preset.get("short_description")
        if description:
            print(f"{'':<44}{description}")


if __name__ == "__main__":
    main()
