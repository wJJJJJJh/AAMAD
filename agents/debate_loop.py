"""Selective routing followed by exactly one exchange per routed candidate."""
from agents.emit_debate import (Prosecutor, Defender, EmitJudge, CONVENTIONAL_PROSECUTOR,
                                CONVENTIONAL_DEFENDER, CONVENTIONAL_JUDGE)
from state import AgentLog
from tools.candidates import distinctive
from tools.evidence import has_data_use, pooled_evidence

def route_candidate(mention) -> str:
    return "ACCEPT" if (distinctive(mention.norm, mention.kind)
                        and has_data_use(pooled_evidence(mention), mention.norm)) else "DEBATE"

def debate_emit_suppress(mentions, bb, llm=None, aligned=True):
    prosecutor = Prosecutor(llm=llm, trace=bb.trace)
    defender = Defender(llm=llm, trace=bb.trace)
    judge = EmitJudge(llm=llm, trace=bb.trace, aligned=aligned)
    if not aligned:
        prosecutor.system = CONVENTIONAL_PROSECUTOR
        defender.system = CONVENTIONAL_DEFENDER
        judge.system = CONVENTIONAL_JUDGE
    retained = []
    for mention in mentions:
        route = route_candidate(mention)
        bb.trace.append(AgentLog("Router", {"dataset_id": mention.norm, "route": route}))
        if route == "ACCEPT":
            retained.append(mention)
            continue
        prosecution = prosecutor.argue(bb, mention)
        defense = defender.argue(bb, mention, prosecution)
        decision = judge.decide(bb, mention, prosecution, defense)
        bb.trace.append(AgentLog("ValidationResult", {"dataset_id": mention.norm, **decision}))
        if decision["decision"] == "EMIT":
            retained.append(mention)
    return retained
