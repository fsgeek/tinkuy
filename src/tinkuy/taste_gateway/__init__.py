"""Taste-native gateway: self-curating tensor in the wire.

The model responds AND curates its own cognitive state. The gateway
injects the tensor, parses updates, and handles the wire. Everything
else is the model's job. Trust it.

No orchestrator. No eviction pipeline. No page table. No cooperative
signals. No proxy paths. The tensor IS the memory. The model IS the
curator. The gateway IS a thin wire with a mirror attached.
"""

from tinkuy.taste_gateway.gateway import (
    MemoryObject,
    TasteGateway,
    TasteGatewayConfig,
    TasteSession,
)

__all__ = ["MemoryObject", "TasteGateway", "TasteGatewayConfig", "TasteSession"]
