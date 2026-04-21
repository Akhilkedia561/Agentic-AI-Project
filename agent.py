"""
agent.py — Physics Study Buddy
Shared agent module used by both the notebook and the Streamlit UI.
Call build_agent() to get (app, embedder, collection).
Call ask(app, question, thread_id) to query the agent.
"""
import os
import re
import math as _math
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Knowledge Base ─────────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": """Newton's Three Laws of Motion form the foundation of classical mechanics.

First Law (Law of Inertia): An object at rest remains at rest, and an object in motion continues in motion with constant velocity, unless acted upon by a net external force. This explains why passengers lurch forward when a bus brakes suddenly — inertia keeps them moving as the bus decelerates.

Second Law (Law of Acceleration): The net force acting on an object equals the product of its mass and acceleration: F = ma. The SI unit of force is the Newton (N), where 1 N = 1 kg·m/s². Example: if a 5 kg object experiences a net force of 20 N, its acceleration is 4 m/s².

Third Law (Action-Reaction): For every action force, there is an equal and opposite reaction force acting on a different object. A rocket expels gas backward (action); the gas pushes the rocket forward (reaction).

Momentum and Impulse: Linear momentum p = mv. Newton's Second Law in general form is F = dp/dt. Impulse J = F·Δt = Δp. When no external forces act, total momentum is conserved: m₁u₁ + m₂u₂ = m₁v₁ + m₂v₂.

Free Body Diagrams: Always isolate the object and draw all forces before applying F = ma. Common forces include weight (W = mg downward), normal force (N perpendicular to surface), friction (f = μN opposing motion), and tension.

Friction: Static friction fs ≤ μsN prevents motion; kinetic friction fk = μkN opposes sliding. Typically μs > μk. On an inclined plane of angle θ: component of gravity along plane = mg·sinθ, normal force = mg·cosθ."""
    },
    {
        "id": "doc_002",
        "topic": "Kinematics — Equations of Motion",
        "text": """Kinematics describes motion without considering its causes.

Key Variables: u = initial velocity (m/s), v = final velocity (m/s), a = uniform acceleration (m/s²), s = displacement (m), t = time (s).

Four Equations of Motion (valid for uniform acceleration):
1. v = u + at
2. s = ut + ½at²
3. v² = u² + 2as
4. s = ½(u + v)t

Free Fall: Under gravity alone, a = g = 9.8 m/s² downward. Taking downward positive: v = gt, s = ½gt². Example: ball dropped from 80 m height — time: 80 = ½(9.8)t², t ≈ 4.04 s; speed at impact v = 9.8 × 4.04 ≈ 39.6 m/s.

Projectile Motion: Motion under gravity with an initial velocity at angle θ. Horizontal and vertical components are independent.
- Horizontal: x = u·cosθ · t  (constant velocity)
- Vertical: y = u·sinθ · t − ½gt²  (uniform deceleration)
- Time of flight: T = 2u·sinθ / g
- Maximum height: H = u²sin²θ / (2g)
- Range: R = u²sin(2θ) / g — maximum range occurs at θ = 45°

Relative Motion: Velocity of A relative to B: v_AB = v_A − v_B.

Velocity-Time Graphs: Slope of v-t graph = acceleration. Area under v-t graph = displacement."""
    },
    {
        "id": "doc_003",
        "topic": "Work, Energy, and Power",
        "text": """Work, energy, and power are central scalar concepts in mechanics.

Work: W = F·d·cosθ, where θ is the angle between force and displacement vectors. Units: Joules (J). Work is positive when force and displacement align, zero when perpendicular, and negative when opposing motion.

Kinetic Energy: KE = ½mv². A 2 kg ball at 10 m/s has KE = ½(2)(10²) = 100 J.

Potential Energy:
- Gravitational PE = mgh  (h measured from reference level)
- Elastic PE = ½kx²  (spring compressed/stretched by x, spring constant k N/m)

Work-Energy Theorem: Net work done on an object equals its change in kinetic energy: W_net = ΔKE = ½mv² − ½mu².

Conservation of Mechanical Energy: When only conservative forces act, KE + PE = constant. A ball dropped from height h hits the ground at v = √(2gh).

Non-Conservative Forces: Friction converts mechanical energy to heat. W_friction = −f·d.

Power: P = W/t = F·v. Unit: Watts (W). A motor lifting 100 kg by 5 m in 10 s: P = (100)(9.8)(5)/10 = 490 W. Horsepower: 1 hp = 746 W.

Efficiency: η = (useful output energy / total input energy) × 100%.

Collisions:
- Elastic: both momentum AND kinetic energy conserved
- Inelastic: only momentum conserved; KE lost to heat/deformation
- Perfectly inelastic: objects stick together; maximum KE loss"""
    },
    {
        "id": "doc_004",
        "topic": "Simple Harmonic Motion",
        "text": """Simple Harmonic Motion (SHM) is periodic oscillation where the restoring force is proportional to displacement and directed toward equilibrium.

Defining condition: F = −kx. Acceleration: a = −ω²x.

Equations of SHM:
- Displacement: x(t) = A·cos(ωt + φ)
- Velocity: v(t) = −Aω·sin(ωt + φ)  →  max speed = Aω at x = 0
- Acceleration: a(t) = −Aω²·cos(ωt + φ)  →  max at x = ±A
- Angular frequency: ω = 2π/T = 2πf

Spring-Mass System:
- ω = √(k/m),   T = 2π√(m/k),   f = (1/2π)√(k/m)
- Example: k = 100 N/m, m = 1 kg → T = 2π√(1/100) ≈ 0.628 s

Simple Pendulum (small angles):
- ω = √(g/L),   T = 2π√(L/g)
- Example: L = 1 m → T ≈ 2.007 s
- Period is independent of mass and amplitude for small oscillations

Energy in SHM:
- KE = ½k(A² − x²)  — maximum at x = 0
- PE = ½kx²  — maximum at x = ±A
- Total energy E = ½kA² = constant

Resonance: Driving frequency = natural frequency → maximum amplitude.
Damping: Underdamped oscillates with decreasing amplitude. Critically damped returns fastest without oscillating."""
    },
    {
        "id": "doc_005",
        "topic": "Thermodynamics — Laws and Processes",
        "text": """Thermodynamics studies heat, work, and energy transformations in systems.

Zeroth Law: If A is in thermal equilibrium with C, and B is in thermal equilibrium with C, then A and B are in thermal equilibrium. This defines temperature.

Ideal Gas Law: PV = nRT, where P = pressure (Pa), V = volume (m³), n = moles, R = 8.314 J/(mol·K), T = Kelvin.
Boyle's Law: PV = constant (isothermal). Charles's Law: V/T = constant (isobaric).

First Law of Thermodynamics: ΔU = Q − W. ΔU = change in internal energy, Q = heat added, W = work done BY system. Conservation of energy.

Thermodynamic Processes:
- Isothermal (T constant): ΔU = 0, Q = W
- Adiabatic (Q = 0): ΔU = −W; PVᵞ = constant
- Isochoric (V constant): W = 0, ΔU = Q
- Isobaric (P constant): W = PΔV

Second Law: Heat flows spontaneously hot → cold. Entropy never decreases. Carnot efficiency: η = 1 − TL/TH.

Specific Heat: Q = mcΔT. Specific heat of water = 4186 J/(kg·K).
Latent Heat: Q = mL. Fusion of water = 334 kJ/kg. Vaporization = 2260 kJ/kg."""
    },
    {
        "id": "doc_006",
        "topic": "Electrostatics — Coulomb's Law and Electric Field",
        "text": """Electrostatics deals with stationary electric charges and their interactions.

Coulomb's Law: F = kq₁q₂/r², where k = 9 × 10⁹ N·m²/C², ε₀ = 8.854 × 10⁻¹² C²/(N·m²). Like charges repel; unlike attract.

Electric Field: E = F/q₀ = kQ/r² for point charge Q. Unit: N/C or V/m. Direction: outward for +Q, inward for −Q.

Superposition Principle: Net field = vector sum of individual fields from all charges.

Electric Potential: V = kQ/r (scalar, Volts). W = qV. E = −dV/dr.

Gauss's Law: Φ = ∮E·dA = Q_enclosed/ε₀. Inside conductor: E = 0. Outside: E = σ/ε₀.

Capacitance: C = Q/V (Farads). Parallel plate: C = ε₀A/d. Energy: U = ½CV².
Series: 1/C_total = 1/C₁ + 1/C₂. Parallel: C_total = C₁ + C₂.

Electric Dipole: ±q separated by 2a. Dipole moment p = q·2a. Torque: τ = pE·sinθ."""
    },
    {
        "id": "doc_007",
        "topic": "Current Electricity — Ohm's Law and Circuits",
        "text": """Current electricity deals with the flow of electric charges through conductors.

Electric Current: I = dQ/dt, in Amperes (A). Conventional current flows from high to low potential.

Ohm's Law: V = IR. V = voltage (V), I = current (A), R = resistance (Ω). Valid for ohmic conductors at constant temperature.

Resistance and Resistivity: R = ρL/A. For metals: ρ increases with temperature: ρ = ρ₀(1 + αT).

Power: P = VI = I²R = V²/R. Energy = Pt.

Kirchhoff's Laws:
- KCL: ΣI_in = ΣI_out at any junction.
- KVL: ΣV = 0 around any closed loop.

Series: R_total = R₁ + R₂ + R₃. Same current; voltage divides proportionally.
Parallel: 1/R_total = 1/R₁ + 1/R₂ + 1/R₃. Same voltage; current divides inversely.

EMF and Internal Resistance: Terminal voltage V = ε − Ir.

Drift Velocity: v_d = I/(neA), where n = electron density, e = 1.6 × 10⁻¹⁹ C. Typical v_d ~ 10⁻⁴ m/s."""
    },
    {
        "id": "doc_008",
        "topic": "Optics — Reflection, Refraction, and Lenses",
        "text": """Optics studies the behavior of light: reflection, refraction, and image formation.

Laws of Reflection: Angle of incidence = angle of reflection (from normal). Rays and normal are coplanar.

Mirror Formula: 1/f = 1/v + 1/u. Magnification: m = −v/u. Concave: f negative. Convex: f positive.

Snell's Law (Refraction): n₁sinθ₁ = n₂sinθ₂. Refractive index n = c/v (c = 3 × 10⁸ m/s). Water: n ≈ 1.33. Glass: n ≈ 1.5.

Total Internal Reflection (TIR): Light from denser to rarer medium at angle > critical angle θ_c. sin(θ_c) = n₂/n₁. Applications: optical fibers, diamonds.

Thin Lens Formula: 1/f = 1/v − 1/u (Cartesian convention). Power P = 1/f (Dioptres).
Convex lens: f positive (converging). Concave: f negative (diverging).

Lens Maker's Equation: 1/f = (n − 1)(1/R₁ − 1/R₂).

Young's Double Slit: Fringe width β = λD/d. Constructive at path diff = nλ. Destructive at (2n+1)λ/2."""
    },
    {
        "id": "doc_009",
        "topic": "Modern Physics — Photoelectric Effect and Bohr Model",
        "text": """Modern physics covers quantum phenomena that classical physics cannot explain.

Photoelectric Effect (Einstein, 1905):
- KE_max = hν − φ. h = 6.626 × 10⁻³⁴ J·s, ν = frequency, φ = work function.
- Threshold frequency: ν₀ = φ/h. Below ν₀ no emission regardless of intensity.
- Stopping potential: eV₀ = KE_max.
- Photon energy: E = hν = hc/λ.

De Broglie Wavelength: λ = h/p = h/(mv). For electron through potential V: λ = h/√(2meV).

Bohr Model of Hydrogen:
- Angular momentum quantized: L = nℏ, n = 1, 2, 3, ...
- Orbital radii: r_n = n²·a₀, a₀ = 0.529 Å (Bohr radius)
- Energy levels: E_n = −13.6/n² eV
  - n=1 (ground state): −13.6 eV
  - n=2: −3.4 eV
  - Ionization energy: 13.6 eV

Spectral Series: Lyman (→n=1, UV), Balmer (→n=2, visible), Paschen (→n=3, IR).
Photon energy for n₂→n₁: E = 13.6(1/n₁² − 1/n₂²) eV.

Heisenberg's Uncertainty Principle: Δx·Δp ≥ ℏ/2."""
    },
    {
        "id": "doc_010",
        "topic": "Gravitation — Newton's Law and Orbital Motion",
        "text": """Gravitation is the attractive force between any two masses.

Newton's Law: F = Gm₁m₂/r². G = 6.674 × 10⁻¹¹ N·m²/kg².

Acceleration due to Gravity: g = GM/R². For Earth: M = 5.97 × 10²⁴ kg, R = 6.37 × 10⁶ m, g ≈ 9.8 m/s². Decreases with altitude: g' = g[R/(R+h)]².

Gravitational Potential Energy: U = −Gm₁m₂/r (U = 0 at ∞).

Escape Velocity: v_escape = √(2GM/R) = √(2gR). For Earth: ≈ 11.2 km/s.

Orbital Velocity: v_orbit = √(GM/r). At Earth's surface: ≈ 7.9 km/s.

Time Period of Orbit: T = 2π√(r³/GM). Low Earth orbit: T ≈ 84 min.

Kepler's Laws:
1. Planets orbit in ellipses, Sun at one focus.
2. Radius vector sweeps equal areas in equal times.
3. T² ∝ a³; T²/a³ = 4π²/(GM) = constant.

Geostationary Orbit: T = 24 h, altitude ≈ 35,786 km above equator."""
    },
    {
        "id": "doc_011",
        "topic": "Rotational Motion — Torque and Moment of Inertia",
        "text": """Rotational motion is the angular analog of linear motion.

Analogies: x↔θ, v↔ω, a↔α, m↔I, F↔τ, p↔L.

Newton's Second Law for Rotation: τ_net = Iα.

Moment of Inertia (about stated axis):
- Solid disc/cylinder (central axis): I = ½MR²
- Hollow cylinder: I = MR²
- Solid sphere (diameter): I = (2/5)MR²
- Hollow sphere (diameter): I = (2/3)MR²
- Thin rod (center, perpendicular): I = ML²/12
- Thin rod (end): I = ML²/3

Parallel Axis Theorem: I = I_cm + Md².

Rotational KE: KE_rot = ½Iω². Rolling without slipping: KE_total = ½mv² + ½Iω².

Angular Momentum: L = Iω. Conserved when net torque = 0. Ice skater example: pulling arms in decreases I, increases ω.

Equations of Rotational Motion (uniform α):
ω = ω₀ + αt;  θ = ω₀t + ½αt²;  ω² = ω₀² + 2αθ."""
    },
    {
        "id": "doc_012",
        "topic": "Waves and Sound",
        "text": """Waves transfer energy without permanent displacement of the medium.

Wave Equation: v = fλ. Period T = 1/f. Angular frequency ω = 2πf. Wave number k = 2π/λ.
Travelling wave: y(x,t) = A·sin(kx − ωt).

Types: Transverse (displacement ⊥ propagation, e.g. light, string). Longitudinal (displacement ∥ propagation, e.g. sound).

Speed of Sound: In air at 20°C: v ≈ 343 m/s. Temperature dependence: v ≈ 331 + 0.6T m/s (T in °C).

Interference:
- Constructive: path difference = nλ → amplitude = 2A
- Destructive: path difference = (2n+1)λ/2 → amplitude = 0

Standing Waves:
- String (both ends fixed) / open pipe: f_n = nv/(2L). Fundamental f₁ = v/(2L).
- Closed pipe: f_n = (2n−1)v/(4L) — odd harmonics only.

Beats: Two frequencies f₁ and f₂ → beat frequency = |f₁ − f₂|.

Doppler Effect: f' = f₀(v + v_observer)/(v − v_source).
Example: 500 Hz source at 30 m/s toward observer → f' = 500×343/(343−30) ≈ 548 Hz."""
    },
]

# ── State ──────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    answer:        str
    faithfulness:  float
    eval_retries:  int
    student_name:  Optional[str]   # extracted from "my name is ..."


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2


# ── build_agent ────────────────────────────────────────────────────────────────
def build_agent():
    """Build and return (compiled_app, embedder, collection)."""
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm_eval = ChatGroq(model="llama-3.1-8b-instant",    temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("physics_kb")
    except Exception:
        pass
    collection = client.create_collection("physics_kb")

    texts      = [d["text"]  for d in DOCUMENTS]
    ids        = [d["id"]    for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ── Node definitions (closures capture llm, embedder, collection) ──────────

    def memory_node(state: CapstoneState) -> dict:
        msgs         = state.get("messages", [])
        question     = state["question"]
        student_name = state.get("student_name")

        msgs = msgs + [{"role": "user", "content": question}]
        if len(msgs) > 6:
            msgs = msgs[-6:]

        # Extract student name if introduced ("my name is Riya")
        name_match = re.search(r"my name is ([A-Za-z]+)", question, re.IGNORECASE)
        if name_match:
            student_name = name_match.group(1).strip().title()

        return {"messages": msgs, "student_name": student_name}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(
            f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
        ) or "none"

        prompt = f"""You are a router for a Physics Study Buddy chatbot for B.Tech students.

Available options:
- retrieve: search the physics knowledge base for conceptual explanations, laws, formulas, and theory.
  ALSO use retrieve when the question contains a wrong or suspicious physical value (e.g., wrong speed of light,
  wrong gravitational constant, incorrect formula) — the knowledge base can verify and correct it.
- memory_only: answer from conversation history (e.g. "what did you just say?", "what formula did you mention?")
- tool: use the physics calculator ONLY when the question asks to numerically compute something
  AND the given values appear physically reasonable (e.g. "calculate the kinetic energy of a 5 kg ball at 10 m/s").
  Do NOT route to tool if the question asserts a physically impossible value (e.g. speed of light as 1000 m/s).

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

        decision = llm_eval.invoke(prompt).content.strip().lower()
        if "memory" in decision:   decision = "memory_only"
        elif "tool" in decision:   decision = "tool"
        else:                      decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        try:
            q_emb   = embedder.encode([state["question"]]).tolist()
            results = collection.query(
                query_embeddings=q_emb, n_results=3,
                include=["documents", "metadatas"]
            )
            chunks  = results["documents"][0]
            topics  = [m["topic"] for m in results["metadatas"][0]]
            context = "\n\n---\n\n".join(
                f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
            )
        except Exception:
            context, topics = "", []
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        extract_prompt = f"""You are a physics calculator. Extract the mathematical expression to compute from this question.
Return ONLY a valid Python expression using numbers and math module functions (math.sqrt, math.pi, math.sin, math.cos, math.exp, math.log).
Do NOT include units. If no numerical computation is possible, return NONE.

Examples:
"kinetic energy of 2 kg at 10 m/s"         -> 0.5 * 2 * 10**2
"period of pendulum length 1 m"             -> 2 * math.pi * math.sqrt(1 / 9.8)
"escape velocity Earth radius 6.37e6 m"     -> math.sqrt(2 * 9.8 * 6.37e6)
"force between 3e-6 C and 4e-6 C at 0.1m"  -> 9e9 * 3e-6 * 4e-6 / 0.1**2

Question: {question}
Expression:"""

        expr = llm.invoke(extract_prompt).content.strip()
        if not expr or expr.upper() == "NONE":
            return {"tool_result": ""}
        try:
            safe_ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
            safe_ns["abs"] = abs
            result = eval(expr, {"__builtins__": {}}, safe_ns)
            tool_result = f"Physics Calculator: {expr} = {result:.6g}"
        except Exception as e:
            tool_result = f"Could not compute '{expr}': {e}"
        return {"tool_result": tool_result}

    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        student_name = state.get("student_name")

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"CALCULATOR RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        name_str = f"The student's name is {student_name}." if student_name else ""

        if context:
            system_content = (
                "You are a Physics Study Buddy assistant for B.Tech students.\n"
                "Answer using ONLY the information provided in the context below.\n"
                "If the answer is not in the context, say: I don't have that specific "
                "information in my knowledge base. Please consult your textbook or ask "
                "your professor.\n"
                "Do NOT add information from your training data. "
                "Do NOT fabricate formulas or physical constants.\n"
                "When a calculator result is provided, incorporate it naturally.\n"
                + (f"{name_str}\n" if name_str else "")
                + "\n" + context
            )
        else:
            system_content = (
                "You are a Physics Study Buddy. Answer based on the conversation history.\n"
                + (name_str if name_str else "")
            )

        if eval_retries > 0:
            system_content += (
                "\n\nIMPORTANT: Your previous answer did not meet quality standards. "
                "Answer using ONLY information explicitly stated in the context above."
            )

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    _REFUSAL_PHRASES = [
        "i don't have that specific information",
        "not in my knowledge base",
        "please consult your textbook",
        "please consult your",
        "i don't have information",
    ]

    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:3000]  # 3000 covers full first retrieved doc
        retries = state.get("eval_retries", 0)

        # Correct refusals are faithful — agent stayed within its knowledge
        if any(p in answer.lower() for p in _REFUSAL_PHRASES):
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = (
            "You are evaluating whether an AI answer is grounded in the provided context.\n"
            "Score faithfulness from 0.0 to 1.0. Reply with ONLY a single decimal number — nothing else.\n\n"
            "Scoring guide:\n"
            "1.0 = every claim in the answer is directly supported by the context\n"
            "0.8 = almost all claims are supported; minor extra detail\n"
            "0.7 = most claims supported; one or two small additions\n"
            "0.5 = some claims unsupported or slightly hallucinated\n"
            "0.0 = answer is fabricated or contradicts the context\n\n"
            f"Context:\n{context}\n\nAnswer:\n{answer[:400]}\n\nFaithfulness score (0.0 to 1.0):"
        )
        raw = llm_eval.invoke(prompt).content.strip()
        try:
            match = re.search(r"\d+\.?\d*", raw)
            score = float(match.group()) if match else 0.8
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.8

        gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
        print(f"  [eval] Faithfulness: {score:.2f} {gate}")
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    # ── Routing functions ──────────────────────────────────────────────────────

    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":        return "tool"
        if route == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")

    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"},
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection


# ── ask() — convenience wrapper for tests and CLI ─────────────────────────────
def ask(app, question: str, thread_id: str = "test-thread") -> dict:
    """
    Query the compiled agent and return the final state dict.
    Uses thread_id for MemorySaver — same thread_id preserves conversation memory.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result
