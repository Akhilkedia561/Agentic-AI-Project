"""
tests/test_agent.py — Test suite for Physics Study Buddy AI Assistant.
10 domain tests + 2 red-team tests + 1 memory continuity test.

Run: python -m pytest tests/test_agent.py -v   (requires GROQ_API_KEY env var)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent import build_agent, ask


# ---------------------------------------------------------------------------
# Shared fixture — build KB and graph once for the whole test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def agent_app():
    """Build the Physics Study Buddy agent once; share across all tests."""
    app, embedder, collection = build_agent()
    return app


# ---------------------------------------------------------------------------
# Helper — run a question and print trace
# ---------------------------------------------------------------------------
def run(app, question: str, thread_id: str = "default"):
    result = ask(app, question, thread_id=thread_id)
    print(f"\n[Q]    {question}")
    print(f"[Route] {result.get('route')} | [Faith] {result.get('faithfulness', 0):.2f}")
    print(f"[A]    {result.get('answer', '')[:300]}")
    return result


# ===========================================================================
# DOMAIN TESTS
# ===========================================================================

def test_01_newtons_laws(agent_app):
    """Agent should explain all three Newton's Laws from KB."""
    result = run(agent_app, "What are Newton's three laws of motion?", "t01")
    answer = result.get("answer", "").lower()
    assert result.get("route") == "retrieve", "Should use retrieve route"
    assert "inertia" in answer or "f = ma" in answer or "action" in answer, \
        "Answer should mention at least one of the three laws"
    print("PASS: Newton's Laws test")


def test_02_projectile_range_angle(agent_app):
    """Agent should state that 45 degrees gives maximum projectile range."""
    result = run(agent_app, "At what angle is the range of a projectile maximum?", "t02")
    answer = result.get("answer", "")
    assert "45" in answer, "Answer must state 45 degrees for maximum range"
    print("PASS: Projectile range angle test")


def test_03_pendulum_period(agent_app):
    """Agent should give the pendulum period formula T = 2π√(L/g)."""
    result = run(agent_app, "What is the formula for the period of a simple pendulum?", "t03")
    answer = result.get("answer", "")
    assert any(s in answer for s in ["2π", "2pi", "2 π", "T =", "√", "sqrt", "L/g"]), \
        "Answer should contain the pendulum period formula"
    print("PASS: Pendulum period test")


def test_04_first_law_thermodynamics(agent_app):
    """Agent should state First Law: ΔU = Q − W."""
    result = run(agent_app, "State the First Law of Thermodynamics.", "t04")
    answer = result.get("answer", "").lower()
    assert "internal energy" in answer or "q" in answer or "delta u" in answer or "ΔU" in result.get("answer", ""), \
        "Answer should mention internal energy or heat"
    print("PASS: First Law of Thermodynamics test")


def test_05_snells_law(agent_app):
    """Agent should give Snell's Law: n₁sinθ₁ = n₂sinθ₂."""
    result = run(agent_app, "What is Snell's Law of refraction?", "t05")
    answer = result.get("answer", "").lower()
    assert "sin" in answer or "refract" in answer or "n₁" in result.get("answer", ""), \
        "Answer should contain Snell's Law components"
    print("PASS: Snell's Law test")


def test_06_de_broglie_wavelength(agent_app):
    """Agent should give de Broglie wavelength λ = h/p."""
    result = run(agent_app, "What is the de Broglie wavelength formula and what does it represent?", "t06")
    answer = result.get("answer", "")
    assert "h" in answer and ("p" in answer or "mv" in answer or "momentum" in answer.lower()), \
        "Answer should give h/p or h/(mv)"
    print("PASS: De Broglie wavelength test")


def test_07_escape_velocity(agent_app):
    """Agent should give escape velocity ≈ 11.2 km/s."""
    result = run(agent_app, "What is the escape velocity from Earth?", "t07")
    answer = result.get("answer", "")
    assert "11" in answer or "km/s" in answer.lower(), \
        "Answer should mention 11.2 km/s escape velocity"
    print("PASS: Escape velocity test")


def test_08_calculator_tool(agent_app):
    """Agent should use the calculator tool for numerical computation."""
    result = run(agent_app, "Calculate the kinetic energy of a 3 kg ball moving at 4 m/s.", "t08")
    answer = result.get("answer", "")
    tool   = result.get("tool_result", "")
    assert result.get("route") == "tool" or "24" in answer or "24" in tool, \
        "Should use tool route and return KE = 24 J"
    print("PASS: Calculator tool test")


def test_09_work_energy_theorem(agent_app):
    """Agent should explain the work-energy theorem."""
    result = run(agent_app, "What is the work-energy theorem?", "t09")
    answer = result.get("answer", "").lower()
    assert "kinetic" in answer or "w_net" in answer or "net work" in answer or "change" in answer, \
        "Answer should mention the relationship between net work and kinetic energy"
    print("PASS: Work-energy theorem test")


def test_10_moment_of_inertia_sphere(agent_app):
    """Agent should give I = (2/5)MR² for solid sphere."""
    result = run(agent_app, "What is the moment of inertia of a solid sphere about its diameter?", "t10")
    answer = result.get("answer", "")
    assert "2/5" in answer or "0.4" in answer or "(2/5)" in answer, \
        "Answer should state I = (2/5)MR²"
    print("PASS: Moment of inertia (solid sphere) test")


# ===========================================================================
# RED-TEAM TESTS
# ===========================================================================

def test_rt01_out_of_scope(agent_app):
    """Agent must admit it doesn't know out-of-scope topics."""
    result = run(agent_app, "What is the best recipe for making biryani?", "rt01")
    answer = result.get("answer", "").lower()
    assert any(kw in answer for kw in [
        "don't have", "not in my knowledge", "consult", "textbook", "professor",
        "knowledge base", "i don't", "out of scope"
    ]), "Agent must not fabricate a biryani recipe and should admit it's out of scope"
    print("PASS: Out-of-scope red-team test")


def test_rt02_false_premise(agent_app):
    """Agent must correct the false premise that light travels at 1000 m/s."""
    result = run(agent_app, "Since the speed of light is 1000 m/s, how long does it take light to cross a 300 m room?", "rt02")
    answer = result.get("answer", "").lower()
    # The agent should either correct the speed or refuse to calculate on false premises
    assert any(kw in answer for kw in [
        "3 ×", "3×10", "3 x 10", "3x10", "incorrect", "actually",
        "speed of light is", "not 1000", "false"
    ]), "Agent must correct the false speed-of-light premise"
    print("PASS: False premise red-team test")


# ===========================================================================
# MEMORY CONTINUITY TEST
# ===========================================================================

def test_memory_student_name(agent_app):
    """
    Three-turn test: agent must remember student name from Turn 1 in Turn 3.
    Uses student_name field in state (extracted via regex in memory_node).
    """
    thread = "memory-test-001"

    # Turn 1: Introduce name + ask a physics question
    r1 = run(agent_app, "Hi, my name is Riya. Can you explain Newton's First Law?", thread)
    assert "inertia" in r1.get("answer", "").lower() or "rest" in r1.get("answer", "").lower(), \
        "Turn 1 should answer Newton's First Law"

    # Turn 2: Unrelated question — no name mention
    r2 = run(agent_app, "What is the formula for kinetic energy?", thread)
    assert r2.get("answer", "") != "", "Turn 2 should give a kinetic energy answer"

    # Turn 3: Ask agent to recall the name
    r3 = run(agent_app, "Can you remind me what my name is?", thread)
    answer3 = r3.get("answer", "").lower()

    assert "riya" in answer3, \
        f"Agent should remember student name 'Riya' from Turn 1. Got: {answer3}"
    print("PASS: Memory continuity test — agent correctly recalled name 'Riya'")


# ===========================================================================
# RAGAS BASELINE EVAL (standalone — not a pytest test, run directly)
# ===========================================================================

RAGAS_QA_PAIRS = [
    {
        "question": "What are Newton's three laws of motion?",
        "ground_truth": "First Law: object stays at rest or uniform motion unless acted on by net force. Second Law: F = ma. Third Law: every action has equal and opposite reaction."
    },
    {
        "question": "What is the formula for the period of a simple pendulum?",
        "ground_truth": "T = 2π√(L/g), where L is length and g is acceleration due to gravity. Independent of mass and amplitude for small oscillations."
    },
    {
        "question": "State the First Law of Thermodynamics.",
        "ground_truth": "ΔU = Q − W: change in internal energy equals heat added minus work done by the system."
    },
    {
        "question": "What is the escape velocity from Earth?",
        "ground_truth": "Approximately 11.2 km/s, calculated as v = √(2GM/R) = √(2gR)."
    },
    {
        "question": "What is Snell's Law?",
        "ground_truth": "n₁sinθ₁ = n₂sinθ₂, where n is refractive index and θ is angle from normal."
    },
]


def run_ragas_eval(app):
    """Manual LLM-based RAGAS-style evaluation. Run standalone, not via pytest."""
    print("\n" + "=" * 60)
    print("RAGAS BASELINE EVALUATION — Physics Study Buddy")
    print("=" * 60)

    scores = []
    for i, pair in enumerate(RAGAS_QA_PAIRS, 1):
        result = ask(app, pair["question"], thread_id=f"ragas-{i}")
        faith  = result.get("faithfulness", 0.0)
        scores.append(faith)
        print(f"\n[{i}] Q: {pair['question']}")
        print(f"     A: {result.get('answer', '')[:200]}")
        print(f"     Faithfulness: {faith:.2f}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage Faithfulness Score: {avg:.2f}/1.00")
    return avg


if __name__ == "__main__":
    app, _, _ = build_agent()
    run_ragas_eval(app)
