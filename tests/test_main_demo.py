"""The demo in ``main.py`` is documentation, so it has to actually run.

Nothing else in the suite executes it: it is not a module the compiler imports
and ruff only looks at the Python around it. It had drifted into three separate
breakages (a function whose body brace had gone missing, a counter assigned
without ``mut``, and a call that cast a string into an ``i64`` parameter), which
is exactly the kind of rot a smoke test catches for free.
"""

from __future__ import annotations

import main


def test_demo_runs_and_prints_its_own_results(capsys):
    main.main()

    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert "hello 65535 1000000.0015" in lines
    assert "sum = 55" in lines
    assert "add = 3" in lines
    assert "fact = 120" in lines
    assert "add(1, 2) = 3" in lines
    assert "factorial(5) = 120" in lines
