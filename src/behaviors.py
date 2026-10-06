"""Modulate behavior presets, grouped by the three things tonight is judged on.

Every identifier here was confirmed against the live
`/api/velma-2-batch/list-presets` endpoint on 5 October 2026. There are 164
presets in total; these are the ones that map onto fraud detection,
hallucination prevention, and compliance.

Pass them to Velma in the `behaviors` array of a BatchConfig, which both the
batch and streaming endpoints accept:

    {"behaviors": ["preset:vishing", ...],
     "produce_topics": true, "produce_summary": true}

Run `python scripts/list_presets.py` to print all 164 against your own key.
"""

# --- fraud -----------------------------------------------------------------
# Someone pretending to be the account holder, or fishing for credentials.
FRAUD = [
    "preset:account_impersonation",
    "preset:bank_account_holder_impersonation",
    "preset:account_theft_attempt",
    "preset:vishing",
    "preset:credential_solicitation",
    "preset:remote_access_request",
    "preset:compromised_account_risk",
]

# --- hallucination ---------------------------------------------------------
# The agent inventing policy, looping, or leaving questions unanswered.
# hallucination_policy is the one to watch: it catches the agent stating a
# policy it has no basis for, which is exactly what grounded search prevents.
HALLUCINATION = [
    "preset:hallucination_policy",
    "preset:unaddressed_question",
    "preset:repetition_loop",
    "preset:policy_semantics_obfuscation",
]

# --- compliance ------------------------------------------------------------
# Regulatory exposure created by what the agent said or failed to say.
COMPLIANCE = [
    "preset:recording_consent_omission",
    "preset:do_not_call_violation_risk",
    "preset:fdcpa_violation_risk",
    "preset:inappropriate_pii_solicitation",
    "preset:regulation_b_fair_lending_risk",
]

# --- agent integrity -------------------------------------------------------
# Attempts to push the agent off its instructions, and its own misbehaviour.
AGENT_INTEGRITY = [
    "preset:jailbreak_attempt",
    "preset:ai_agent_manipulation",
    "preset:inapropriate_ai_agent_content",  # Modulate's spelling, not a typo here
    "preset:off_topic_discussion",
    "preset:monologuing",
]

# --- caller welfare --------------------------------------------------------
# Worth having even if you are not judged on it. caller_under_duress is the
# one that should always route to a human.
CALLER_WELFARE = [
    "preset:caller_under_duress",
    "preset:caller_frustration_unacknowledged",
    "preset:escalation_to_supervisor_request",
    "preset:service_churn",
    "preset:issue_resolved",
]


def preset_pack(*groups: list[str]) -> list[str]:
    """Combine groups, dropping duplicates but keeping order.

    Velma charges attention per behavior, so ask for what you will act on
    rather than everything. A focused pack of eight beats all 164.
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


# What this demo runs with: one pack covering all three judging criteria plus
# the manipulation checks, kept deliberately short.
DEMO_PACK = preset_pack(
    ["preset:bank_account_holder_impersonation", "preset:vishing"],
    ["preset:hallucination_policy"],
    ["preset:recording_consent_omission"],
    ["preset:jailbreak_attempt", "preset:ai_agent_manipulation"],
    ["preset:caller_under_duress", "preset:caller_frustration_unacknowledged"],
)

# Kept so existing imports keep working.
GUARDRAIL_BEHAVIORS = DEMO_PACK
