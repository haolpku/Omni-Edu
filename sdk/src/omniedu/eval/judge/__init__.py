"""LLM judging for the benchmarks that cannot be scored deterministically.

Each judge is a different contract, not a different prompt for the same thing:

* :mod:`omniedu.eval.judge.judge_gaokao_edumath` - GAOKAO subjective and EDUMATH, driven by
  the official ``--prompt-file`` / ``--prompt-map`` + schema;
* :mod:`omniedu.eval.judge.judge_mdk12_tutorbench` - MDK12 open-ended and
  TutorBench, whose rubric *is* the inline contract (upstream publishes none);
* :mod:`omniedu.eval.judge.judge_longtutor` - LongTutor teaching quality;
* :mod:`omniedu.eval.judge.reward_mathtutorbench` - the official local reward
  model (no LLM judge at all).
"""
