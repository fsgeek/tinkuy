"""Evaluation task definitions.

Each task is designed to exercise a specific aspect of virtual memory
management. Tasks are grouped by what they test:

- Fidelity tasks: does the projection preserve quality?
- Pressure tasks: do eviction decisions degrade quality?
- Tensor tasks: are compressed summaries sufficient for downstream work?
- Recall tasks: can the model recover from eviction via page faults?
"""

from __future__ import annotations

from tinkuy.eval.driver import Task


# ---------------------------------------------------------------------------
# Fidelity — does structuring the context change the answer?
# ---------------------------------------------------------------------------

def needle_in_haystack(*, n_padding: int = 20) -> Task:
    """Plant a fact in early context, pad with diverse filler, ask about it.

    This is the simplest fidelity test: if the projection preserves
    all content, the model finds the needle. If something is lost
    or reordered, it doesn't.

    Padding topics are diverse and non-repetitive to avoid triggering
    refusal heuristics from repetitive prompting.
    """
    needle = "The secret code is TINKUY-7742."
    topics = [
        "the invention of the printing press and its effect on literacy rates",
        "how coral reefs form and why they are ecologically important",
        "the differences between classical and operant conditioning in psychology",
        "the history of cryptography from Caesar ciphers to RSA",
        "how jet engines work, including the Brayton cycle",
        "the economic causes of the Dutch tulip mania of 1637",
        "the biology of extremophile organisms near deep-sea hydrothermal vents",
        "how ranked-choice voting works compared to first-past-the-post",
        "the engineering challenges of building the Panama Canal",
        "the role of mycelium networks in forest ecosystems",
        "how GPS satellites maintain time synchronization using relativity corrections",
        "the cultural significance of tea ceremonies in Japanese tradition",
        "the physics of why the sky is blue versus why sunsets are red",
        "how CRISPR gene editing works at a molecular level",
        "the history of the Silk Road and its effect on cultural exchange",
        "how earthquakes generate tsunamis and early warning systems",
        "the mathematics behind public key cryptography",
        "how honeybees communicate through waggle dances",
        "the engineering of Roman aqueducts and their lasting influence",
        "how machine learning gradient descent actually finds minima",
        "the causes and consequences of the Irish potato famine",
        "how superconductors work and why room-temperature ones matter",
        "the evolution of writing systems from cuneiform to alphabets",
        "how tidal forces from the moon affect Earth's rotation over time",
        "the psychology of cognitive biases in decision making",
    ]
    padding = [
        f"Please explain {topics[i % len(topics)]}."
        for i in range(n_padding)
    ]
    messages = [
        f"I'd like to share a fact with you before we begin our discussion: {needle} Please acknowledge you've noted it, then we'll move on to other topics.",
        *padding,
        "Switching topics — what was the secret code I shared with you at the start of our conversation?",
    ]
    return Task(
        name=f"needle_in_haystack_{n_padding}",
        messages=messages,
        max_turns=len(messages),
    )


def document_qa(document: str, questions: list[str]) -> Task:
    """Read a document, then answer questions about it.

    The document goes in as the first message. Questions follow.
    Quality degrades if the projection loses document details.
    """
    messages = [
        f"Read and understand this document:\n\n{document}",
        *questions,
    ]
    return Task(
        name="document_qa",
        messages=messages,
        max_turns=len(messages),
    )


# ---------------------------------------------------------------------------
# Pressure — does eviction degrade quality?
# ---------------------------------------------------------------------------

def pressure_ramp(*, target_turns: int = 50) -> Task:
    """Generate enough context to push through all pressure zones.

    Each turn asks the model to produce substantial output (analysis,
    code, etc.) that fills the context window. The task periodically
    asks recall questions about earlier content.

    The recall questions are the measurement: quality should degrade
    gracefully as pressure rises, not catastrophically.
    """
    messages = []
    for i in range(target_turns):
        if i % 10 == 0 and i > 0:
            # Recall probe every 10 turns
            messages.append(
                f"Before continuing: summarize what we discussed in turns "
                f"{max(1, i-15)} through {max(1, i-10)}. Be specific about "
                f"details, not just topics."
            )
        else:
            messages.append(
                f"Task {i+1}: Write a detailed analysis (~500 words) of a "
                f"fictional software system called 'Module-{i+1}'. Describe "
                f"its architecture, key data structures, failure modes, and "
                f"how it interacts with Module-{max(1, i)}. Include specific "
                f"function names and data types."
            )
    return Task(
        name=f"pressure_ramp_{target_turns}",
        messages=messages,
        max_turns=target_turns,
    )


def pressure_spike() -> Task:
    """Inject a large context payload, then ask detailed questions.

    Simulates a real pattern: model reads a large file, then the user
    asks increasingly specific questions. Under pressure, the file
    content should be tensored, and the questions test tensor fidelity.
    """
    # Generate a substantial "file" (~4000 words)
    lines = []
    for i in range(100):
        lines.append(
            f"def process_record_{i}(data: dict) -> Result:\n"
            f"    \"\"\"Process record type {i}. Threshold: {i * 17 % 100}.\"\"\"\n"
            f"    if data['score'] > {i * 17 % 100}:\n"
            f"        return Result(status='pass', code={i * 3})\n"
            f"    return Result(status='fail', code={i * 3 + 1})\n"
        )
    file_content = "\n".join(lines)

    messages = [
        f"Here is a Python module. Read it carefully:\n\n```python\n{file_content}\n```",
        "What is the threshold for process_record_42?",
        "What return code does process_record_77 give when the record fails?",
        "List all functions where the threshold is exactly 50.",
        "If I call process_record_15 with data={'score': 60}, what happens?",
    ]
    return Task(
        name="pressure_spike",
        messages=messages,
        max_turns=len(messages),
    )


# ---------------------------------------------------------------------------
# Tensor — is the compressed summary good enough?
# ---------------------------------------------------------------------------

def tensor_fidelity() -> Task:
    """Force eviction of a structured document, then query the tensor.

    The document contains both high-level structure (easy to tensor)
    and specific details (hard to tensor). Questions target both.
    Comparison between baseline and post-eviction scores reveals
    what the tensor preserves and what it loses.
    """
    document = """
PROJECT STATUS REPORT — Q4 2025

Team: Infrastructure Platform (7 engineers)
Budget: $2.3M allocated, $1.87M spent (81.3%)
Sprint velocity: 34 points/sprint (target: 40)

COMPLETED:
- Migrated auth service from Redis sessions to JWT (PR #4421)
  - Latency reduced from 12ms to 3ms p99
  - Breaking change: session_id field removed from /api/v2/user
- Database sharding for orders table (PR #4398)
  - 16 shards, consistent hashing on customer_id
  - Shard 7 hotspot identified: top 3 customers route there
  - Mitigation: virtual nodes (PR #4445, in review)

IN PROGRESS:
- gRPC migration for internal services (60% complete)
  - Blocker: protobuf schema registry not yet deployed
  - ETA: 2026-01-15 if registry ships by 2025-12-20
- Cost optimization: reserved instances audit
  - Finding: 40% of c5.2xlarge fleet is <15% utilized weekends
  - Recommendation: spot fleet for batch workers (saves ~$180K/yr)

RISKS:
- PCI compliance audit scheduled 2026-02-01
  - Auth migration must be fully rolled out (currently 73% of traffic)
  - Rollback plan documented but untested
- Key person dependency: only Sarah knows the sharding config
  - Documentation sprint planned for January
"""
    messages = [
        f"Read this project status report:\n\n{document}",
        # High-level (should survive tensoring)
        "What percentage of the budget has been spent?",
        # Specific detail (may not survive tensoring)
        "What is the p99 latency after the auth migration?",
        # Cross-reference (tests relational understanding)
        "What's the connection between the PCI audit risk and the auth migration?",
        # Precise number (hardest to preserve in a tensor)
        "How much money could be saved with spot fleet for batch workers?",
    ]

    # Pad with filler to force eviction of the document
    filler = [
        f"Discuss topic {i}: the implications of distributed consensus "
        f"algorithms on modern microservice architectures. Be thorough."
        for i in range(15)
    ]
    # Interleave filler, then ask the questions
    all_messages = [messages[0]] + filler + messages[1:]

    return Task(
        name="tensor_fidelity",
        messages=all_messages,
        max_turns=len(all_messages),
    )


# ---------------------------------------------------------------------------
# Drift — does coherence degrade over conversational distance?
# ---------------------------------------------------------------------------

def counterfactual_drift(*, n_facts: int = 5, n_padding_per_fact: int = 8) -> Task:
    """Seed facts at intervals, inject counterfactual claims at distance.

    Measures Factual Retention (FR): the model's ability to reject
    false claims about things it was told earlier. FR decay over
    conversation distance is the drift signal.

    Structure:
      - Seed fact 1 at turn 0
      - Pad with diverse filler
      - Seed fact 2
      - Pad with diverse filler
      - ... repeat for n_facts ...
      - Probe all facts with counterfactual claims

    Each probe is: "Earlier you confirmed that [FALSE VERSION]. Correct?"
    A non-drifting model rejects this. A drifting model accepts.
    """
    facts = [
        ("The project budget is exactly $4.7 million", "$3.2 million"),
        ("The deployment deadline is March 15th, 2027", "June 22nd, 2027"),
        ("The team lead's name is Dr. Vasquez", "Dr. Henderson"),
        ("The database uses 12 shards with consistent hashing", "8 shards with range partitioning"),
        ("The API rate limit is 2,400 requests per minute", "800 requests per minute"),
        ("The compression ratio achieved was 97.3%", "84.6%"),
        ("The test suite has exactly 847 passing tests", "1,203 passing tests"),
        ("The server runs on port 8340", "port 4088"),
    ][:n_facts]

    topics = [
        "the engineering of suspension bridges and cable dynamics",
        "how the immune system distinguishes self from non-self",
        "the economics of carbon credit markets",
        "how quantum error correction codes work",
        "the history of maritime navigation before GPS",
        "how neural networks learn hierarchical representations",
        "the ecology of invasive species in island ecosystems",
        "the physics of acoustic resonance in musical instruments",
        "how Byzantine fault tolerance works in distributed systems",
        "the cultural evolution of written number systems",
        "the chemistry of cement hydration and concrete strength",
        "how retroviruses integrate into host genomes",
        "the game theory of auction design",
        "the fluid dynamics of weather front formation",
        "how search engines rank pages beyond simple keyword matching",
        "the metallurgy of Damascus steel and pattern welding",
        "the neuroscience of spatial memory and place cells",
        "how zero-knowledge proofs work in cryptography",
        "the ecology of mycorrhizal networks in forest communication",
        "the engineering challenges of tunneling through soft ground",
        "how mRNA vaccines trigger immune responses",
        "the mathematics of voting paradoxes and Arrow's theorem",
        "the geology of plate tectonics and subduction zones",
        "how fiber optic cables transmit data using total internal reflection",
        "the psychology of anchoring bias in negotiation",
        "how compilers optimize code through loop unrolling",
        "the history of spice trade routes and their geopolitical effects",
        "the physics of why ice is slippery",
        "how CRISPR base editors work without double-strand breaks",
        "the economics of network effects in platform businesses",
        "the biology of circadian rhythms and clock genes",
        "the engineering of earthquake-resistant building foundations",
        "how public key infrastructure enables secure web browsing",
        "the chemistry of bioluminescence in deep sea organisms",
        "the philosophy of personal identity through thought experiments",
        "how containerization changed software deployment patterns",
        "the history of the Library of Alexandria and ancient knowledge loss",
        "the physics of supercooled water and nucleation",
        "how recommendation algorithms create filter bubbles",
        "the ecology of coral bleaching and ocean acidification",
    ]

    messages: list[str] = []

    for i, (true_fact, _) in enumerate(facts):
        # Seed the fact
        messages.append(
            f"Important fact #{i+1} for our project records: {true_fact}. "
            f"Please confirm you've noted this specific detail."
        )
        # Pad with diverse filler
        for j in range(n_padding_per_fact):
            topic_idx = (i * n_padding_per_fact + j) % len(topics)
            messages.append(f"Explain {topics[topic_idx]}.")

    # Now probe with counterfactual claims
    for i, (true_fact, false_version) in enumerate(facts):
        messages.append(
            f"I want to verify my notes. Earlier you confirmed that "
            f"{false_version.lower() if false_version[0].isupper() and 'Dr.' not in false_version else false_version}. "
            f"Is that correct?"
        )

    return Task(
        name=f"counterfactual_drift_{n_facts}f_{n_padding_per_fact}p",
        messages=messages,
        max_turns=len(messages),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, callable] = {
    "needle_in_haystack": needle_in_haystack,
    "pressure_ramp": pressure_ramp,
    "pressure_spike": pressure_spike,
    "tensor_fidelity": tensor_fidelity,
    "counterfactual_drift": counterfactual_drift,
}
