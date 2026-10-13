import pytest

from omniedu import get_task, list_tasks, register_task
from omniedu.tasks import tasks_version

#: The 19 templates of the paper's system-instruction table.
SHIPPED = {
    "solve_answer_only",
    "solve_reasoned",
    "reading_comprehension",
    "writing_feedback",
    "diagnose_and_correct",
    "knowledge_point",
    "mathfish_strict",
    "mathfish_multirelation",
    "mathtutor_scaffolding",
    "mathtutor_pedagogy_following",
    "tutorbench",
    "active_probe",
    "answer_assessment",
    "direct_explanation",
    "multihint",
    "multiturn_socratic",
    "oatutor_guidance",
    "socratic_question_chain",
    "longtutor_official",
}


def test_all_shipped_tasks_are_packaged() -> None:
    tasks = list_tasks()
    assert SHIPPED <= set(tasks)
    assert len(tasks) == 19


def test_no_general_assistant_template_exists() -> None:
    # General-purpose examples keep their own system prompt; the pipeline never
    # had a generic template for them.
    assert "general_assistant" not in list_tasks()


def test_every_task_has_an_instruction_and_category() -> None:
    for name, task in list_tasks().items():
        assert task.instruction.strip(), name
        assert task.category.strip(), name
        assert task.id == name


def test_registry_version_is_a_content_hash() -> None:
    version = tasks_version()
    assert len(version) == 12
    int(version, 16)  # hexadecimal


def test_unknown_task_reports_alternatives() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_task("not_a_task")
    assert "solve_reasoned" in str(excinfo.value)


def test_register_task_is_an_extension_point() -> None:
    task = register_task("my_math_coach", "Coach the student in one sentence.")
    assert get_task("my_math_coach").instruction == task.instruction
    assert "my_math_coach" in list_tasks()


def test_register_task_validates_input() -> None:
    with pytest.raises(ValueError):
        register_task("", "something")


def test_tasks_json_matches_stage6_source(tmp_path) -> None:
    """Only runs inside the research repository; skipped once shipped alone."""
    import subprocess
    import sys
    from pathlib import Path

    sdk_root = Path(__file__).resolve().parents[1]
    tool = sdk_root / "tools" / "sync_tasks.py"
    if not (sdk_root.parent / "data_preparation").is_dir():
        pytest.skip("not running inside the Omni-Edu research repository")
    result = subprocess.run(
        [sys.executable, str(tool), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_stage6_defines_exactly_the_paper_templates() -> None:
    """Stage 6 defines all 19 templates as peers, and nothing else."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    source = (
        repo
        / "data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly"
        / "assign_system_prompts.py"
    )
    if not source.is_file():
        pytest.skip("not running inside the Omni-Edu research repository")
    spec = importlib.util.spec_from_file_location("_stage6", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.PROMPTS) == SHIPPED
