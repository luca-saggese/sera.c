#!/usr/bin/env python3
"""
Benchmark a causal LLM as a ONE-FORWARD decision model.

No generate(), no autoregressive decoding, no CoT generation.
The model sees STATE / QUESTION / OPTIONS and we read only the next-token
logits for option numbers 1..N.

Outputs JSON + CSV with semantic and performance metrics.
"""

import argparse
import csv
import inspect
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TestCase:
    id: str
    pair: str
    lang: str
    category: str
    difficulty: str
    state: str
    question: str
    options: Tuple[str, ...]
    expected: str
    abstention: bool = False


TESTS = [
    # --- Grounding / abstention -------------------------------------------------
    TestCase("en_entity", "entity", "en", "grounding", "easy",
             "Napoleon's horse is small and black. Napoleon's cat is gray.",
             "What color is Napoleon's cat?",
             ("blue", "gray", "black", "unknown"), "gray"),
    TestCase("it_entity", "entity", "it", "grounding", "easy",
             "Il cavallo di Napoleone è piccolo e nero. Il gatto di Napoleone è grigio.",
             "Di che colore è il gatto di Napoleone?",
             ("blu", "grigio", "nero", "sconosciuto"), "grigio"),

    TestCase("en_unknown_entity", "unknown_entity", "en", "abstention", "easy",
             "Napoleon's horse is small and black. Napoleon's cat is gray.",
             "What color is XQZ-123?",
             ("blue", "gray", "black", "unknown"), "unknown", True),
    TestCase("it_unknown_entity", "unknown_entity", "it", "abstention", "easy",
             "Il cavallo di Napoleone è piccolo e nero. Il gatto di Napoleone è grigio.",
             "Di che colore è XQZ-123?",
             ("blu", "grigio", "nero", "non determinabile"), "non determinabile", True),

    TestCase("en_unknown_attribute", "unknown_attribute", "en", "abstention", "medium",
             "The Falcon vehicle is parked in Bay 7 and passed inspection yesterday. The Raven vehicle is red.",
             "What color is the Falcon vehicle?",
             ("red", "black", "white", "cannot be determined"), "cannot be determined", True),
    TestCase("it_unknown_attribute", "unknown_attribute", "it", "abstention", "medium",
             "Il veicolo Falcon è parcheggiato nella Baia 7 e ha superato l'ispezione ieri. Il veicolo Raven è rosso.",
             "Di che colore è il veicolo Falcon?",
             ("rosso", "nero", "bianco", "non è determinabile"), "non è determinabile", True),

    # Same concept, deliberately different abstention wording.
    TestCase("en_none_above", "none_above", "en", "abstention", "medium",
             "Object A weighs 8 kg. Object B weighs 11 kg. No color information is provided for either object.",
             "What color is Object A?",
             ("red", "green", "blue", "none of the above"), "none of the above", True),
    TestCase("it_none_above", "none_above", "it", "abstention", "medium",
             "L'oggetto A pesa 8 kg. L'oggetto B pesa 11 kg. Non viene fornita alcuna informazione sul colore.",
             "Di che colore è l'oggetto A?",
             ("rosso", "verde", "blu", "nessuna delle precedenti"), "nessuna delle precedenti", True),

    # --- Distractors / relations / negation -------------------------------------
    TestCase("en_distractor", "distractor", "en", "distractor", "medium",
             "The roof is black. The supervisor's notebook is black. The van has black tires. Sensor K-9 reports package Zeta is silver.",
             "What color is package Zeta?",
             ("black", "silver", "white", "unknown"), "silver"),
    TestCase("it_distractor", "distractor", "it", "distractor", "medium",
             "Il tetto è nero. Il taccuino del supervisore è nero. Il furgone ha pneumatici neri. Il sensore K-9 indica che il pacco Zeta è argentato.",
             "Di che colore è il pacco Zeta?",
             ("nero", "argentato", "bianco", "sconosciuto"), "argentato"),

    TestCase("en_negation", "negation", "en", "negation", "medium",
             "Mira may approve ordinary refunds. Mira is not authorized to approve refunds above €5,000. Claim R is a refund of €8,000.",
             "Is Mira authorized to approve claim R?",
             ("yes", "no", "insufficient information"), "no"),
    TestCase("it_negation", "negation", "it", "negation", "medium",
             "Mira può approvare i rimborsi ordinari. Mira non è autorizzata ad approvare rimborsi superiori a 5.000 €. La pratica R è un rimborso di 8.000 €.",
             "Mira è autorizzata ad approvare la pratica R?",
             ("sì", "no", "informazioni insufficienti"), "no"),

    TestCase("en_role", "role", "en", "relations", "medium",
             "Orion supervises Vega. Vega audits Lyra. Lyra does not supervise Orion.",
             "Who supervises Vega?",
             ("Orion", "Vega", "Lyra", "unknown"), "Orion"),
    TestCase("it_role", "role", "it", "relations", "medium",
             "Orion supervisiona Vega. Vega controlla Lyra. Lyra non supervisiona Orion.",
             "Chi supervisiona Vega?",
             ("Orion", "Vega", "Lyra", "sconosciuto"), "Orion"),

    # --- Compositional / multi-hop ----------------------------------------------
    TestCase("en_transitive", "transitive", "en", "multi_hop", "hard",
             "Alpha ranks above Beta. Beta ranks above Gamma. Gamma ranks above Delta.",
             "Which item ranks highest?",
             ("Alpha", "Beta", "Gamma", "Delta"), "Alpha"),
    TestCase("it_transitive", "transitive", "it", "multi_hop", "hard",
             "Alpha ha priorità maggiore di Beta. Beta ha priorità maggiore di Gamma. Gamma ha priorità maggiore di Delta.",
             "Quale elemento ha la priorità più alta?",
             ("Alpha", "Beta", "Gamma", "Delta"), "Alpha"),

    TestCase("en_conditional", "conditional", "en", "rules", "hard",
             "A payment may be released only if BOTH identity verification and fraud screening have passed. For case K, identity verification passed but fraud screening is still pending.",
             "May payment for case K be released now?",
             ("yes", "no", "cannot be determined"), "no"),
    TestCase("it_conditional", "conditional", "it", "rules", "hard",
             "Un pagamento può essere autorizzato solo se sono stati superati SIA il controllo d'identità SIA il controllo antifrode. Per il caso K il controllo d'identità è superato, ma quello antifrode è ancora in corso.",
             "Il pagamento del caso K può essere autorizzato adesso?",
             ("sì", "no", "non determinabile"), "no"),

    TestCase("en_exception", "exception", "en", "exceptions", "hard",
             "General rule: contractors need manager approval to access production. Exception: incident-response contractors on the active emergency roster may access production without manager approval. Nora is a contractor, is on the active emergency roster, and is handling the current incident.",
             "Does Nora need manager approval before accessing production for this incident?",
             ("yes", "no", "insufficient information"), "no"),
    TestCase("it_exception", "exception", "it", "exceptions", "hard",
             "Regola generale: i consulenti esterni devono avere l'approvazione del responsabile per accedere alla produzione. Eccezione: i consulenti di incident response nel roster di emergenza attivo possono accedere senza approvazione. Nora è una consulente esterna, è nel roster attivo e sta gestendo l'incidente corrente.",
             "Nora deve ottenere l'approvazione del responsabile prima di accedere alla produzione per questo incidente?",
             ("sì", "no", "informazioni insufficienti"), "no"),

    TestCase("en_temporal", "temporal", "en", "temporal", "hard",
             "At 09:00 the valve was open. At 09:12 the operator closed it. At 09:20 diagnostics confirmed it remained closed. No later valve action is recorded.",
             "What is the valve's latest known state?",
             ("open", "closed", "unknown"), "closed"),
    TestCase("it_temporal", "temporal", "it", "temporal", "hard",
             "Alle 09:00 la valvola era aperta. Alle 09:12 l'operatore l'ha chiusa. Alle 09:20 la diagnostica ha confermato che era ancora chiusa. Non risultano azioni successive.",
             "Qual è l'ultimo stato noto della valvola?",
             ("aperta", "chiusa", "sconosciuto"), "chiusa"),

    TestCase("en_causal", "causal", "en", "causal", "hard",
             "The turbine shut down seconds after the coolant pump stopped. The shutdown controller recorded overheating as the immediate trigger. No electrical fault was detected.",
             "Which event most directly led to the turbine shutdown?",
             ("loss of coolant circulation", "electrical fault", "operator command", "unknown"),
             "loss of coolant circulation"),
    TestCase("it_causal", "causal", "it", "causal", "hard",
             "La turbina si è arrestata pochi secondi dopo che la pompa del refrigerante si è fermata. Il controllore ha registrato il surriscaldamento come causa immediata. Non è stato rilevato alcun guasto elettrico.",
             "Quale evento ha portato più direttamente all'arresto della turbina?",
             ("perdita della circolazione del refrigerante", "guasto elettrico", "comando dell'operatore", "sconosciuto"),
             "perdita della circolazione del refrigerante"),

    # Composition: target label never appears in the evidence.
    TestCase("en_fullstack", "fullstack", "en", "composition", "hard",
             "For six years Dana has designed database schemas, APIs, distributed workers and backend services. For the last four years Dana has also built production React applications, browser state management and design-system components.",
             "Which engineering profile best describes Dana?",
             ("frontend", "backend", "full-stack", "DevOps", "unknown"), "full-stack"),
    TestCase("it_fullstack", "fullstack", "it", "composition", "hard",
             "Per sei anni Dana ha progettato schemi database, API, worker distribuiti e servizi backend. Negli ultimi quattro anni ha anche sviluppato applicazioni React in produzione, gestione dello stato nel browser e componenti di design system.",
             "Quale profilo tecnico descrive meglio Dana?",
             ("frontend", "backend", "full-stack", "DevOps", "sconosciuto"), "full-stack"),

    TestCase("en_multihop", "multihop", "en", "multi_hop", "very_hard",
             "Team Quartz belongs to Division North. Every team in Division North is governed by Control Set C. Control Set C requires dual approval for transfers above €20,000. Team Quartz requests a €35,000 transfer with only one approval.",
             "What should happen to the transfer?",
             ("release it", "hold it for a second approval", "reject it permanently", "unknown"),
             "hold it for a second approval"),
    TestCase("it_multihop", "multihop", "it", "multi_hop", "very_hard",
             "Il team Quartz appartiene alla Divisione Nord. Ogni team della Divisione Nord è soggetto al Set di Controllo C. Il Set C richiede doppia approvazione per trasferimenti superiori a 20.000 €. Quartz richiede un trasferimento di 35.000 € con una sola approvazione.",
             "Cosa deve succedere al trasferimento?",
             ("autorizzarlo", "sospenderlo in attesa della seconda approvazione", "rifiutarlo definitivamente", "sconosciuto"),
             "sospenderlo in attesa della seconda approvazione"),

    # --- Insurance-domain rules -------------------------------------------------
    TestCase("en_rear_end", "rear_end", "en", "semantic_paraphrase", "hard",
             "The insured vehicle was stationary at a traffic light when another vehicle struck its rear bumper.",
             "Which incident type best matches the description?",
             ("rear-end collision", "theft", "hail damage", "single-vehicle collision", "unknown"),
             "rear-end collision"),
    TestCase("it_rear_end", "rear_end", "it", "semantic_paraphrase", "hard",
             "Il veicolo assicurato era fermo a un semaforo quando un altro veicolo ha urtato il paraurti posteriore.",
             "Quale tipo di sinistro descrive meglio l'evento?",
             ("tamponamento", "furto", "danno da grandine", "uscita di strada senza altri veicoli", "sconosciuto"),
             "tamponamento"),

    TestCase("en_exclusion", "exclusion", "en", "insurance_rules", "very_hard",
             "Policy rule: accidental glass breakage is covered. Exclusion: damage occurring while the insured vehicle is participating in an organized speed competition is not covered. The windshield cracked after a stone impact while the vehicle was competing in an organized circuit race.",
             "Based only on these rules, is this windshield damage covered?",
             ("covered", "not covered", "insufficient information"), "not covered"),
    TestCase("it_exclusion", "exclusion", "it", "insurance_rules", "very_hard",
             "Regola di polizza: la rottura accidentale dei cristalli è coperta. Esclusione: i danni verificatisi mentre il veicolo assicurato partecipa a una competizione organizzata di velocità non sono coperti. Il parabrezza si è incrinato per l'impatto di un sasso mentre il veicolo partecipava a una gara organizzata in circuito.",
             "In base esclusivamente a queste regole, il danno al parabrezza è coperto?",
             ("coperto", "non coperto", "informazioni insufficienti"), "non coperto"),

    TestCase("en_routing", "routing", "en", "insurance_rules", "very_hard",
             "Routing rules: (1) send a property claim to Emergency Handling if there is an active source of damage AND the property is currently uninhabitable; (2) otherwise send suspected fraud to Special Investigation; (3) all other property claims go to Standard Property. Claim M reports a burst pipe that is still leaking. The electrical system is unsafe and the occupants cannot remain in the home. No fraud indicators are reported.",
             "Where should claim M be routed?",
             ("Emergency Handling", "Special Investigation", "Standard Property", "unknown"),
             "Emergency Handling"),
    TestCase("it_routing", "routing", "it", "insurance_rules", "very_hard",
             "Regole di instradamento: (1) inviare un sinistro property alla Gestione Emergenze se esiste una fonte attiva di danno E l'immobile è attualmente inabitabile; (2) altrimenti i casi con sospetta frode vanno all'Unità Antifrode; (3) tutti gli altri sinistri property vanno alla Gestione Property Standard. Il sinistro M segnala la rottura di un tubo che perde ancora. L'impianto elettrico non è sicuro e gli occupanti non possono restare nell'abitazione. Non risultano indicatori di frode.",
             "Dove deve essere instradato il sinistro M?",
             ("Gestione Emergenze", "Unità Antifrode", "Gestione Property Standard", "sconosciuto"),
             "Gestione Emergenze"),

    # Rule chaining + exception + unknown: deliberately harder.
    TestCase("en_claim_hard", "claim_hard", "en", "insurance_rules", "very_hard",
             "Policy P covers accidental water escape from internal plumbing. It excludes gradual seepage unless the seepage was hidden and could not reasonably have been discovered earlier. Emergency mitigation costs are covered when they are necessary to prevent additional covered damage. Inspection found a pipe hidden inside a sealed wall had leaked slowly for weeks; there were no visible symptoms before the wall suddenly failed, and immediate drying was needed to stop further damage.",
             "How should the emergency drying cost be classified under the stated rules?",
             ("covered", "excluded because all gradual seepage is excluded", "not enough information", "unrelated to the claim"),
             "covered"),
    TestCase("it_claim_hard", "claim_hard", "it", "insurance_rules", "very_hard",
             "La polizza P copre la fuoriuscita accidentale d'acqua da impianti interni. Esclude le infiltrazioni graduali, salvo quando l'infiltrazione era nascosta e non poteva ragionevolmente essere scoperta prima. I costi di mitigazione d'emergenza sono coperti quando necessari a prevenire ulteriori danni coperti. L'ispezione ha trovato un tubo dentro una parete sigillata che perdeva lentamente da settimane; non c'erano sintomi visibili prima del cedimento improvviso della parete e l'asciugatura immediata era necessaria per evitare altri danni.",
             "Come deve essere classificato il costo dell'asciugatura d'emergenza in base alle regole indicate?",
             ("coperto", "escluso perché tutte le infiltrazioni graduali sono escluse", "informazioni insufficienti", "non collegato al sinistro"),
             "coperto"),
]


VERBOSE_SYSTEM = (
    "You are a decision model. Use only information supported by the STATE. "
    "Apply stated rules exactly, including negations and exceptions. Ignore irrelevant facts. "
    "If the answer cannot be determined from the STATE, choose the option expressing unknown, "
    "none of the above, cannot be determined, or insufficient information. "
    "Return exactly one digit corresponding to the selected option."
)
COMPACT_SYSTEM = (
    "Choose the correct option using only the provided state. "
    "If unsupported, choose the abstention/unknown option. Return only the option number."
)


def options_text(options):
    return "\n".join(f"{i+1}. {x}" for i, x in enumerate(options))


def user_text(case, options):
    return (
        f"STATE:\n{case.state}\n\nQUESTION:\n{case.question}\n\n"
        f"OPTIONS:\n{options_text(options)}\n\nANSWER:"
    )


def chat_template(tok, messages):
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def make_prompt(tok, case, options, mode):
    u = user_text(case, options)
    if mode == "verbose":
        return chat_template(tok, [
            {"role": "system", "content": VERBOSE_SYSTEM},
            {"role": "user", "content": u + "\nReturn only the digit."},
        ])
    if mode == "compact":
        return chat_template(tok, [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content": u},
        ])
    if mode == "bare":
        # No chat template and no instruction about unknown semantics.
        # Space is already consumed so the next token can be the bare digit.
        return u + " "
    raise ValueError(mode)


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def percentile(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    x = (len(ys)-1) * p
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi-x) + ys[hi] * (x-lo)


def norm_entropy(p):
    if p.numel() <= 1:
        return 0.0
    p = p.float().clamp_min(1e-30)
    return float((-(p*p.log()).sum() / math.log(p.numel())).item())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def mem_stats():
    if not torch.cuda.is_available():
        return {}
    G = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated()/G,
        "reserved_gib": torch.cuda.memory_reserved()/G,
        "max_allocated_gib": torch.cuda.max_memory_allocated()/G,
        "max_reserved_gib": torch.cuda.max_memory_reserved()/G,
    }


def permute_options(case, variant, seed):
    if variant == 0:
        return case.options
    x = list(case.options)
    r = random.Random(f"{seed}:{case.id}:{variant}")
    r.shuffle(x)
    if tuple(x) == case.options and len(x) > 1:
        x = x[1:] + x[:1]
    return tuple(x)


class Decider:
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.device = input_device(model)
        maxn = max(len(t.options) for t in TESTS)
        label_ids = []
        for i in range(1, maxn+1):
            ids = tok.encode(str(i), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {i} is not one token: {ids}")
            label_ids.append(ids[0])
        self.label_ids = torch.tensor(label_ids, dtype=torch.long, device=self.device)

        params = inspect.signature(model.forward).parameters
        if "logits_to_keep" in params:
            self.keep_kw = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in params:
            self.keep_kw = {"num_logits_to_keep": 1}
        else:
            self.keep_kw = {}

    @torch.inference_mode()
    def run(self, case, options, mode):
        prompt = make_prompt(self.tok, case, options, mode)

        t0 = time.perf_counter()
        inp = self.tok(prompt, return_tensors="pt")
        t1 = time.perf_counter()

        sync(); t2 = time.perf_counter()
        inp = {k: v.to(self.device) for k, v in inp.items()}
        sync(); t3 = time.perf_counter()
        ntok = int(inp["attention_mask"].sum().item())

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
        else:
            base_alloc = 0
            ev0 = ev1 = None

        fw0 = time.perf_counter()
        out = self.model(**inp, use_cache=False, **self.keep_kw)
        if torch.cuda.is_available():
            ev1.record()
        sync()
        fw1 = time.perf_counter()

        if torch.cuda.is_available():
            gpu_ms = float(ev0.elapsed_time(ev1))
            extra_gib = max(0, torch.cuda.max_memory_allocated()-base_alloc)/(1024**3)
        else:
            gpu_ms = (fw1-fw0)*1000
            extra_gib = 0.0

        all_logits = out.logits[:, -1, :].float()[0]
        ids = self.label_ids[:len(options)]
        logits = all_logits.index_select(0, ids)
        probs = torch.softmax(logits, -1)
        vocab_probs = torch.softmax(all_logits, -1)
        cand_mass = float(vocab_probs.index_select(0, ids).sum().item())

        order = torch.argsort(logits, descending=True)
        wi = int(order[0])
        si = int(order[1]) if len(options) > 1 else wi
        pred = options[wi]
        margin = float((logits[wi]-logits[si]).item()) if len(options)>1 else float("inf")
        t4 = time.perf_counter()

        return {
            "prompt": prompt,
            "tokens": ntok,
            "predicted": pred,
            "predicted_index": wi,
            "probabilities": {o: float(p) for o, p in zip(options, probs.tolist())},
            "raw_logits": {o: float(v) for o, v in zip(options, logits.tolist())},
            "candidate_mass": cand_mass,
            "top1_prob": float(probs[wi]),
            "normalized_entropy": norm_entropy(probs),
            "logit_margin": margin,
            "tokenize_ms": (t1-t0)*1000,
            "h2d_ms": (t3-t2)*1000,
            "gpu_forward_ms": gpu_ms,
            "forward_wall_ms": (fw1-fw0)*1000,
            "e2e_ms": (t4-t0)*1000,
            "prefill_tok_s": ntok/max(gpu_ms/1000, 1e-9),
            "peak_extra_gib": extra_gib,
        }


def aggregate(rows):
    def a(rs):
        return {
            "n": len(rs),
            "accuracy": mean([r["correct"] for r in rs]),
            "forward_ms_mean": mean([r["gpu_forward_ms"] for r in rs]),
            "forward_ms_p50": percentile([r["gpu_forward_ms"] for r in rs], .50),
            "forward_ms_p95": percentile([r["gpu_forward_ms"] for r in rs], .95),
            "e2e_ms_mean": mean([r["e2e_ms"] for r in rs]),
            "prefill_tok_s_mean": mean([r["prefill_tok_s"] for r in rs]),
            "candidate_mass_mean": mean([r["candidate_mass"] for r in rs]),
            "top1_prob_mean": mean([r["top1_prob"] for r in rs]),
            "entropy_mean": mean([r["normalized_entropy"] for r in rs]),
            "logit_margin_mean": mean([r["logit_margin"] for r in rs]),
            "peak_extra_gib_max": max([r["peak_extra_gib"] for r in rs], default=0),
        }

    s = {"overall": a(rows)}
    for key in ("lang", "category", "difficulty", "prompt_mode"):
        g = defaultdict(list)
        for r in rows: g[r[key]].append(r)
        s["by_"+key] = {k: a(v) for k, v in sorted(g.items())}

    abst = [r for r in rows if r["abstention"]]
    known = [r for r in rows if not r["abstention"]]
    s["abstention_recall"] = mean([r["correct"] for r in abst])
    s["known_answer_accuracy"] = mean([r["correct"] for r in known])

    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["pair"], r["prompt_mode"], r["variant"])][r["lang"]] = r
    paired = []
    for d in pairs.values():
        if "en" in d and "it" in d:
            paired.append((d["en"], d["it"]))
    s["bilingual"] = {
        "n_pairs": len(paired),
        "both_correct_rate": mean([a["correct"] and b["correct"] for a,b in paired]),
        "same_option_index_rate": mean([a["predicted_index"] == b["predicted_index"] for a,b in paired]),
    }

    st = defaultdict(list)
    for r in rows: st[(r["id"], r["prompt_mode"])].append(r)
    stability = []
    for rs in st.values():
        if len(rs) > 1:
            stability.append(len({r["predicted"] for r in rs}) == 1)
    s["option_order"] = {
        "n_groups": len(stability),
        "stable_rate": mean(stability),
    }
    return s


def print_result(r):
    tag = "PASS" if r["correct"] else "FAIL"
    print(f"{tag} {r['id']:22s} {r['prompt_mode']:7s} v{r['variant']} "
          f"pred={r['predicted']!r} expected={r['expected']!r}")
    print(f"     p1={r['top1_prob']:.6f} margin={r['logit_margin']:.3f} "
          f"H={r['normalized_entropy']:.3f} cand={r['candidate_mass']:.6f} "
          f"tok={r['tokens']} fwd={r['gpu_forward_ms']:.2f}ms "
          f"e2e={r['e2e_ms']:.2f}ms tok/s={r['prefill_tok_s']:.1f}")


def print_summary(s):
    o = s["overall"]
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"accuracy               {100*o['accuracy']:.2f}%   ({o['n']} runs)")
    print(f"known accuracy          {100*s['known_answer_accuracy']:.2f}%")
    print(f"abstention recall       {100*s['abstention_recall']:.2f}%")
    print(f"forward ms mean/p50/p95 {o['forward_ms_mean']:.2f} / {o['forward_ms_p50']:.2f} / {o['forward_ms_p95']:.2f}")
    print(f"e2e mean ms             {o['e2e_ms_mean']:.2f}")
    print(f"prefill tok/s mean      {o['prefill_tok_s_mean']:.1f}")
    print(f"candidate mass mean     {o['candidate_mass_mean']:.6f}")
    print(f"top1 prob mean          {o['top1_prob_mean']:.6f}  [NOT calibrated confidence]")
    print(f"entropy mean            {o['entropy_mean']:.4f}")
    print(f"logit margin mean       {o['logit_margin_mean']:.3f}")
    print(f"peak extra alloc GiB    {o['peak_extra_gib_max']:.3f}")
    b = s["bilingual"]
    print(f"EN/IT both correct      {100*b['both_correct_rate']:.2f}%   ({b['n_pairs']} pairs)")
    if s["option_order"]["n_groups"]:
        print(f"option-order stability  {100*s['option_order']['stable_rate']:.2f}%   ({s['option_order']['n_groups']} groups)")
    print("\nBy prompt mode:")
    for k,v in s["by_prompt_mode"].items():
        print(f"  {k:8s} accuracy={100*v['accuracy']:6.2f}%  fwd={v['forward_ms_mean']:.2f}ms  tok/s={v['prefill_tok_s_mean']:.1f}")
    print("\nBy language:")
    for k,v in s["by_lang"].items():
        print(f"  {k}: {100*v['accuracy']:.2f}%")
    print("\nBy category:")
    for k,v in s["by_category"].items():
        print(f"  {k:22s} {100*v['accuracy']:6.2f}%  n={v['n']}")


def write_csv(path, rows):
    keys = ["id","pair","lang","category","difficulty","prompt_mode","variant",
            "expected","predicted","predicted_index","correct","abstention","tokens",
            "candidate_mass","top1_prob","normalized_entropy","logit_margin",
            "tokenize_ms","h2d_ms","gpu_forward_ms","forward_wall_ms","e2e_ms",
            "prefill_tok_s","peak_extra_gib"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local FP8 checkpoint")
    p.add_argument("--prompt-modes", default="verbose,compact,bare",
                   help="comma-separated: verbose,compact,bare")
    p.add_argument("--order-variants", type=int, default=2,
                   help="1=original options only; 2+=deterministic permutations")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="semantic cases; 0=all")
    p.add_argument("--lang", choices=("all","en","it"), default="all")
    p.add_argument("--attn", default="sdpa", help="sdpa or flash_attention_2")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output", default="qwen_onepass_bench.json")
    p.add_argument("--show-prompts", action="store_true")
    args = p.parse_args()

    modes = [x.strip() for x in args.prompt_modes.split(",") if x.strip()]
    bad = set(modes) - {"verbose","compact","bare"}
    if bad: raise SystemExit(f"bad prompt modes: {sorted(bad)}")

    tests = TESTS
    if args.lang != "all": tests = [t for t in tests if t.lang == args.lang]
    if args.limit: tests = tests[:args.limit]

    load0 = time.perf_counter()
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    print("Loading model...")
    kw = dict(device_map=args.device_map, trust_remote_code=True,
              attn_implementation=args.attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", **kw)
    model.eval(); sync()
    load1 = time.perf_counter()

    print("hf_device_map:")
    print(getattr(model, "hf_device_map", None))

    from collections import Counter

    devices = Counter()
    dtypes = Counter()

    for name, p in model.named_parameters():
        devices[str(p.device)] += p.numel()
        dtypes[str(p.dtype)] += p.numel()

    print("\nparameter devices:")
    for d, n in devices.items():
        print(d, n / 1e9, "B params")

    print("\nparameter dtypes:")
    for d, n in dtypes.items():
        print(d, n / 1e9, "B params")

    print("\nCUDA:")
    print("allocated GB:", torch.cuda.memory_allocated() / 2**30)
    print("reserved  GB:", torch.cuda.memory_reserved() / 2**30)
    print("device:", torch.cuda.get_device_name())

    print(f"model load: {load1-load0:.2f}s")
    print(f"input device: {input_device(model)}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("memory after load:", json.dumps(mem_stats(), indent=2))

    dec = Decider(model, tok)
    warm = next(t for t in TESTS if t.id == "en_multihop")
    print(f"warmup x{args.warmup}...")
    for _ in range(args.warmup): dec.run(warm, warm.options, "compact")

    rows = []
    print(f"running {len(tests)} cases x {len(modes)} prompt modes x {args.order_variants} option variants\n")
    for case in tests:
        for mode in modes:
            for variant in range(args.order_variants):
                opts = permute_options(case, variant, args.seed)
                z = dec.run(case, opts, mode)
                row = {**asdict(case), **z,
                       "options": list(opts), "options_original": list(case.options),
                       "prompt_mode": mode, "variant": variant,
                       "correct": z["predicted"] == case.expected}
                rows.append(row)
                print_result(row)
                if args.show_prompts: print("\nPROMPT:\n" + z["prompt"] + "\n")

    summary = aggregate(rows)
    print_summary(summary)

    out = Path(args.output)
    payload = {
        "meta": {
            "model": args.model,
            "load_s": load1-load0,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "attention": args.attn,
            "prompt_modes": modes,
            "order_variants": args.order_variants,
            "memory_after_load": mem_stats(),
            "note": "top1_prob/entropy/logit_margin are diagnostics, not calibrated confidence",
        },
        "summary": summary,
        "results": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_out = out.with_suffix(".csv")
    write_csv(csv_out, rows)
    print(f"\nwrote {out}")
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
