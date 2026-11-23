"""The bundled ``stdlib/`` modules, exercised on both backends."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from compiler.core.ast_nodes import FnDecl, StructDecl
from compiler.core.include_expander import IncludeNotFoundError, load_program

ROOT = Path(__file__).resolve().parent.parent
STDLIB = ROOT / "stdlib"

MODULES = (
    "math",
    "algo",
    "str_utils",
    "collections",
    "numpy",
    "linalg",
    "random",
    "prelude",
)


# ---------------------------------------------------------------------------
# Probe programs
# ---------------------------------------------------------------------------
#
# Each probe returns a small integer score. The same text is compiled by the
# native backend (via ``SOURCES`` below) and wrapped in an include-expanded
# program for the Python backend.


PROBE_MATH = """\
fn stdlib_probe_math() -> i64 {
    let a = abs_i64(-5);
    let b = min_i64(3, 7);
    let c = max_i64(3, 7);
    let d = clamp_i64(15, 0, 10);
    let e = gcd(48, 18);
    let f = lcm(4, 6);
    let g = pow_mod(2, 10, 1000);
    let p = is_prime(97) as i64;
    let q = is_prime(91) as i64;
    let r = match sqrt_f64(9.0) { Ok(v) => v as i64, Err(_) => -1 };
    let s = exp_f64(0.0) as i64;
    let t = abs_f64(-2.5) as i64;
    a + b + c + d + e + f + g + p * 100 + q + r + s + t
}
"""

PROBE_ALGO = """\
fn stdlib_probe_algo() -> i64 {
    let xs: Vec<i64> = [3, 1, 2, 1].to_vec();
    let sorted = bubble_sort(xs.clone());
    let ins = insertion_sort(xs.clone());
    let sel = selection_sort(xs.clone());
    let pos = match binary_search(sorted.clone(), 2) { Some(p) => p as i64, None => -1 };
    let miss = match binary_search(sorted.clone(), 9) { Some(_) => 1, None => 0 };
    let lin = match linear_search(xs.clone(), 1) { Some(p) => p as i64, None => -1 };
    let total = sum(sorted.clone());
    let maxv = match max_value(xs.clone()) { Some(v) => v, None => 0 };
    let minv = match min_value(xs.clone()) { Some(v) => v, None => 0 };
    let uniq = unique(xs.clone());
    let reps = count_occurrences(xs.clone(), 1);
    let rev = reverse_values(xs.clone());
    let sorted_ok = is_sorted(sorted.clone()) as i64;
    sorted[0] * 1000000
        + ins[0] * 100000
        + sel[0] * 10000
        + (pos + 1) * 1000
        + miss * 100
        + (lin + 1) * 10
        + total
        + maxv
        + minv
        + (uniq.len() as i64)
        + reps
        + rev[0]
        + sorted_ok
}
"""

PROBE_STR_UTILS = """\
fn stdlib_probe_str_utils() -> i64 {
    let s = "  Hello World  ".to_string();
    let t = trim(s.clone());
    let up = to_upper(t.clone());
    let lo = to_lower(t.clone());
    let pre = starts_with(t.clone(), "Hello".to_string()) as i64;
    let suf = ends_with(t.clone(), "World".to_string()) as i64;
    let has = contains(t.clone(), "lo W".to_string()) as i64;
    let parts = split("a,b,c".to_string(), ",".to_string());
    let joined = join(parts.clone(), "-".to_string());
    let padded = pad_left("7".to_string(), 4, "0".to_string());
    let right = pad_right("7".to_string(), 4, "0".to_string());
    let counted = count_substring("banana".to_string(), "an".to_string());
    let bytes = count_byte("banana".to_string(), "a".to_string());
    let pal = is_palindrome("racecar".to_string()) as i64;
    let back = reverse_text("abc".to_string());
    let rep = repeat("ab".to_string(), 3);
    let rs = replace_all("a-b-c".to_string(), "-".to_string(), "+".to_string());
    let ts = trim_start("  x ".to_string());
    let te = trim_end("  x ".to_string());
    (t.len() as i64) * 1000000
        + (up.len() as i64) * 100000
        + (lo.len() as i64) * 10000
        + pre * 1000
        + suf * 100
        + has * 10
        + (parts.len() as i64)
        + (joined.len() as i64)
        + (padded.len() as i64)
        + (right.len() as i64)
        + counted
        + bytes
        + pal
        + (back.len() as i64)
        + (rep.len() as i64)
        + (rs.len() as i64)
        + (ts.len() as i64)
        + (te.len() as i64)
}
"""

PROBE_COLLECTIONS = """\
fn stdlib_probe_collections() -> i64 {
    let mut stack = Stack::new();
    stack.push(10);
    stack.push(20);
    let top = match stack.peek() { Some(v) => v, None => 0 };
    let popped = match stack.pop() { Some(v) => v, None => 0 };
    let stack_len = stack.len() as i64;

    let mut queue = Queue::new();
    queue.enqueue(1);
    queue.enqueue(2);
    let front = match queue.dequeue() { Some(v) => v, None => 0 };
    let queue_len = queue.len() as i64;

    let mut set = SetI64::new();
    let first = set.insert(5) as i64;
    let second = set.insert(5) as i64;
    let set_has = set.contains(5) as i64;
    let removed = set.remove(5) as i64;
    let set_len = set.len() as i64;

    let mut counter = Counter::new();
    counter.add("a".to_string());
    counter.add("a".to_string());
    counter.add("b".to_string());
    let a_count = counter.get("a".to_string());
    let b_count = counter.get("b".to_string());
    let total_count = counter.total();
    let counter_keys = counter.keys().len() as i64;

    top * 10000000
        + popped * 1000000
        + stack_len * 100000
        + front * 10000
        + queue_len * 1000
        + first * 100
        + second * 10
        + set_has
        + removed
        + set_len
        + a_count
        + b_count
        + total_count
        + counter_keys
}
"""

PROBE_NUMPY = """\
fn stdlib_probe_numpy() -> i64 {
    let a: NdArray = match from_vec([1.0, 2.0, 3.0, 4.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let b = eye(2);
    let z = zeros(2, 3);
    let o = ones(2, 2);
    let f = full(2, 2, 7.0);
    let shape = a.shape();
    let get_ok = match a.get(1, 0) { Some(v) => v, None => -1.0 };
    let get_miss = match a.get(9, 9) { Some(_) => 1, None => 0 };
    let mut m = zeros(2, 2);
    let set_ok = m.set(0, 1, 5.0) as i64;
    let at_value = m.at(0, 1);
    let sumab: NdArray = match a.add(&b) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let subab: NdArray = match a.sub(&b) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let mulab: NdArray = match a.mul(&a) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let trace = match a.trace() { Ok(v) => v, Err(_) => -1.0 };
    let total = a.sum();
    let mean = a.mean();
    let norm = a.frobenius_norm();
    let minv = match a.min() { Some(v) => v, None => -1.0 };
    let maxv = match a.max() { Some(v) => v, None => -1.0 };
    let amax = match a.argmax() { Some(v) => v as i64, None => -1 };
    let amin = match a.argmin() { Some(v) => v as i64, None => -1 };
    let t = a.transpose();
    let prod: NdArray = match a.matmul(&a) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let dot: NdArray = match a.dot(&b) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let exp0 = a.exp().at(0, 0);
    let sqrt4 = match a.sqrt() { Ok(v) => v.at(1, 1), Err(_) => -1.0 };
    let absv = a.neg().abs().at(0, 0);
    let scaled = a.scale(2.0).at(0, 0);
    let shifted = a.add_scalar(1.0).at(0, 0);
    let row = match a.row(0) { Ok(v) => v, Err(_) => [].to_vec() };
    let col = match a.col(1) { Ok(v) => v, Err(_) => [].to_vec() };
    let reshaped: NdArray = match a.reshape(4, 1) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let slice = from_slice([1.0, 2.0, 3.0].to_vec());
    let eq = a.approx_eq(&a, 1e-9) as i64;
    (shape[0] as i64) * 1000000000
        + (shape[1] as i64) * 100000000
        + (z.size() as i64) * 10000000
        + (o.sum() as i64) * 1000000
        + (f.sum() as i64) * 100000
        + (get_ok as i64) * 10000
        + get_miss * 1000
        + set_ok * 100
        + (at_value as i64) * 10
        + (sumab.at(0, 0) as i64)
        + (subab.at(0, 0) as i64)
        + (mulab.at(0, 0) as i64)
        + (trace as i64)
        + (total as i64)
        + (mean as i64)
        + (norm as i64)
        + (minv as i64)
        + (maxv as i64)
        + amax
        + amin
        + (t.at(0, 1) as i64)
        + (t.at(1, 0) as i64)
        + (prod.at(0, 0) as i64)
        + (prod.at(1, 1) as i64)
        + (dot.at(0, 0) as i64)
        + (exp0 as i64)
        + (sqrt4 as i64)
        + (absv as i64)
        + (scaled as i64)
        + (shifted as i64)
        + (row.len() as i64)
        + (col.len() as i64)
        + (reshaped.size() as i64)
        + (slice.ncols() as i64)
        + eq
}
"""

PROBE_LINALG = """\
fn stdlib_probe_linalg() -> i64 {
    let a = match from_vec([4.0, 7.0, 2.0, 6.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let inverse = match a.inv() { Ok(v) => v, Err(_) => zeros(0, 0) };
    let identity_check = match a.matmul(&inverse) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let inv_ok = identity_check.approx_eq(&eye(2), 1e-9) as i64;

    let solver = match from_vec([2.0, 1.0, 1.0, 3.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let rhs = match from_vec([5.0, 10.0].to_vec(), 2, 1) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let x = match solver.solve(&rhs) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let free_x = match solve(&solver, &rhs) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let free_inv = match inv(&solver) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let det = match determinant(&solver) { Ok(v) => v, Err(_) => -1.0 };
    let singular = match from_vec([1.0, 2.0, 2.0, 4.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let singular_err = match singular.inv() { Ok(_) => 0, Err(_) => 1 };

    inv_ok * 1000000000
        + (x.at(0, 0) as i64) * 100000000
        + (x.at(1, 0) as i64) * 10000000
        + (free_x.at(0, 0) as i64) * 1000000
        + (free_x.at(1, 0) as i64) * 100000
        + (free_inv.at(0, 0) as i64) * 10000
        + (det as i64) * 1000
        + singular_err * 100
        + (identity_check.at(0, 0) as i64)
}
"""

PROBE_RANDOM = """\
fn stdlib_probe_random() -> i64 {
    let mut r1 = Rng::seed(42u64);
    let mut r2 = Rng::from_seed(42u64);
    let v1 = r1.next_u64();
    let v2 = r2.next_u64();
    let same = (v1 == v2) as i64;
    let seq_ok = (v1 == 1546998764402558742u64) as i64;
    let f = r1.next_f64();
    let f_ok = ((f - 0.3789802506626686) < 1e-15 && (0.3789802506626686 - f) < 1e-15) as i64;
    let b = r1.next_bool() as i64;
    let roll = r1.roll_d6();
    let ranged = r1.range_i64(1, 7);
    let shuffled = r1.shuffle([1, 2, 3, 4, 5].to_vec());
    let shuffled_ok = (shuffled.len() == 5 && shuffled[0] == 4 && shuffled[4] == 5) as i64;
    let mut seeded = seed_rng(42u64);
    same * 1000000
        + seq_ok * 100000
        + f_ok * 10000
        + b * 1000
        + roll * 100
        + ranged * 10
        + shuffled_ok
        + (seeded.next_u64() == 1546998764402558742u64) as i64
}
"""

PROBE_PRELUDE = """\
fn stdlib_probe_prelude() -> i64 {
    let xs: Vec<i64> = [3, 1, 2].to_vec();
    let sorted = bubble_sort(xs.clone());
    let m: NdArray = match from_vec([1.0, 2.0, 3.0, 4.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let inv = match m.inv() { Ok(v) => v, Err(_) => zeros(0, 0) };
    let ident = match m.matmul(&inv) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let mut rng = Rng::seed(42u64);
    let r = rng.next_u64();
    gcd(48, 18) * 1000000
        + sorted[0] * 100000
        + (trim("  x  ".to_string()).len() as i64) * 10000
        + (ident.approx_eq(&eye(2), 1e-9) as i64) * 1000
        + (r == 1546998764402558742u64) as i64 * 100
        + (is_palindrome("racecar".to_string()) as i64) * 10
        + (pow_mod(2, 10, 1000) as i64)
}
"""

PROBE_INV_IDENTITY = """\
fn stdlib_probe_inv_identity() -> i64 {
    let a = match from_vec([4.0, 7.0, 2.0, 6.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let inverse = match a.inv() { Ok(v) => v, Err(_) => zeros(0, 0) };
    let product = match a.matmul(&inverse) { Ok(v) => v, Err(_) => zeros(0, 0) };
    product.approx_eq(&eye(2), 1e-9) as i64
}
"""

PROBE_SOLVE_EXACT = """\
fn stdlib_probe_solve_exact() -> i64 {
    let a = match from_vec([2.0, 1.0, 1.0, 3.0].to_vec(), 2, 2) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let rhs = match from_vec([5.0, 10.0].to_vec(), 2, 1) { Ok(v) => v, Err(_) => zeros(0, 0) };
    let x = match solve(&a, &rhs) { Ok(v) => v, Err(_) => zeros(0, 0) };
    ((x.at(0, 0) == 1.0) && (x.at(1, 0) == 3.0)) as i64
}
"""

PROBE_RNG_SEQUENCE = """\
fn stdlib_probe_rng_sequence() -> i64 {
    let expected: Vec<u64> = [
        1546998764402558742u64,
        6990951692964543102u64,
        12544586762248559009u64,
        17057574109182124193u64,
        18295552978065317476u64,
        14199186830065750584u64,
    ].to_vec();
    let mut rng = Rng::seed(42u64);
    let mut i = 0;
    let mut ok = 1;
    while i < expected.len() {
        if rng.next_u64() != expected[i] { ok = 0; }
        i = i + 1;
    }
    ok
}
"""


# (probe function name, std module for the Python run, source)
PROBES = (
    ("stdlib_probe_math", "math", PROBE_MATH),
    ("stdlib_probe_algo", "algo", PROBE_ALGO),
    ("stdlib_probe_str_utils", "str_utils", PROBE_STR_UTILS),
    ("stdlib_probe_collections", "collections", PROBE_COLLECTIONS),
    ("stdlib_probe_numpy", "numpy", PROBE_NUMPY),
    ("stdlib_probe_linalg", "linalg", PROBE_LINALG),
    ("stdlib_probe_random", "random", PROBE_RANDOM),
    ("stdlib_probe_prelude", "prelude", PROBE_PRELUDE),
    ("stdlib_probe_inv_identity", "linalg", PROBE_INV_IDENTITY),
    ("stdlib_probe_solve_exact", "linalg", PROBE_SOLVE_EXACT),
    ("stdlib_probe_rng_sequence", "random", PROBE_RNG_SEQUENCE),
)

EXPECTED = {
    "stdlib_probe_math": 173,
    "stdlib_probe_algo": 1113038,
    "stdlib_probe_str_utils": 12211151,
    "stdlib_probe_collections": 220111110,
    "stdlib_probe_numpy": 2266830239,
    "stdlib_probe_linalg": 1131305101,
    "stdlib_probe_random": 1111651,
    "stdlib_probe_prelude": 6111134,
    "stdlib_probe_inv_identity": 1,
    "stdlib_probe_solve_exact": 1,
    "stdlib_probe_rng_sequence": 1,
}


def _stdlib_preamble() -> str:
    """All stdlib modules as one include-free source for a single native build."""
    chunks: list[str] = []
    for name in (
        "math",
        "algo",
        "str_utils",
        "collections",
        "numpy",
        "linalg",
        "random",
    ):
        text = (STDLIB / f"{name}.poly").read_text(encoding="utf-8")
        chunks.append(
            "\n".join(
                line
                for line in text.splitlines()
                if not line.strip().startswith("include!(")
            )
        )
    return "\n\n".join(chunks)


PREAMBLE = _stdlib_preamble()
SOURCES = {name: source for name, _module, source in PROBES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_python(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "stdlib_probe.poly"
    path.write_text(source, encoding="utf-8")
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-m", "api.cli", "run", str(path)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _python_program(module: str, probe: str, source: str) -> str:
    return f'include!("std:{module}");\n\n{source}\nfn main() -> i64 {{ {probe}() }}\n'


# ---------------------------------------------------------------------------
# Packaging and resolution
# ---------------------------------------------------------------------------


def test_stdlib_modules_are_bundled():
    for name in MODULES:
        assert (STDLIB / f"{name}.poly").is_file(), name
    assert (STDLIB / "__init__.py").is_file()


def test_stdlib_is_declared_in_pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    tools = config["tool"]["setuptools"]
    assert "stdlib*" in tools["packages"]["find"]["include"]
    assert tools["package-data"]["stdlib"] == ["*.poly"]


@pytest.mark.parametrize(
    "raw", ["std:missing_module", "std:../math", "std:/etc/passwd"]
)
def test_bad_std_include_is_e006(tmp_path: Path, raw: str):
    main = tmp_path / "main.poly"
    main.write_text(f'include!("{raw}");\nfn main() -> i64 {{ 0 }}\n', encoding="utf-8")
    with pytest.raises(IncludeNotFoundError) as excinfo:
        load_program(main)
    assert excinfo.value.code == "E006"


def test_prelude_defines_ndarray_once(tmp_path: Path):
    main = tmp_path / "main.poly"
    main.write_text(
        'include!("std:numpy");\ninclude!("std:prelude");\nfn main() -> i64 { 0 }\n',
        encoding="utf-8",
    )
    program = load_program(main)
    structs = [item.name for item in program.items if isinstance(item, StructDecl)]
    fns = [item.name for item in program.items if isinstance(item, FnDecl)]
    assert structs.count("NdArray") == 1
    assert fns.count("sqrt_f64") == 1
    assert fns.count("inv") == 1


def test_linalg_and_numpy_define_ndarray_once(tmp_path: Path):
    main = tmp_path / "main.poly"
    main.write_text(
        'include!("std:linalg");\ninclude!("std:numpy");\nfn main() -> i64 { 0 }\n',
        encoding="utf-8",
    )
    program = load_program(main)
    structs = [item.name for item in program.items if isinstance(item, StructDecl)]
    assert structs.count("NdArray") == 1


# ---------------------------------------------------------------------------
# Python backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("probe", "module", "source"), PROBES)
def test_stdlib_python_backend(tmp_path: Path, probe: str, module: str, source: str):
    result = _run_python(tmp_path, _python_program(module, probe, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(EXPECTED[probe])


def test_stdlib_python_backend_dedups_numpy_and_prelude(tmp_path: Path):
    source = (
        'include!("std:numpy");\n'
        'include!("std:prelude");\n\n'
        f"{PROBE_PRELUDE}\n"
        "fn main() -> i64 { stdlib_probe_prelude() }\n"
    )
    result = _run_python(tmp_path, source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(EXPECTED["stdlib_probe_prelude"])


# ---------------------------------------------------------------------------
# Native backend
# ---------------------------------------------------------------------------


@pytest.mark.native
@pytest.mark.parametrize(("probe", "_module", "_source"), PROBES)
def test_stdlib_native_backend(native_ns, probe: str, _module: str, _source: str):
    assert native_ns[probe]() == EXPECTED[probe]


# ---------------------------------------------------------------------------
# Added pure helpers (str_utils / algo)
# ---------------------------------------------------------------------------


def _extra_module_ns(tmp_path: Path, module: str) -> dict:
    from backends.python.translator import compile_rust

    path = tmp_path / f"{module}_extra.poly"
    path.write_text(
        f'include!("std:{module}");\nfn main() -> i64 {{ 0 }}\n',
        encoding="utf-8",
    )
    return compile_rust(load_program(path))


def test_str_utils_added_helpers_python_backend(tmp_path: Path):
    ns = _extra_module_ns(tmp_path, "str_utils")

    assert ns["trim_matches"]("xxabcxx", "xy") == "abc"
    assert ns["replace_first"]("a-b-c", "-", "+") == "a+b-c"
    assert ns["replace_first"]("abc", "-", "+") == "abc"

    remainder = ns["split_once"]("a:b:c", ":")
    assert remainder.is_some()
    assert remainder.unwrap() == "b:c"
    assert ns["split_once"]("abc", ":").is_none()

    assert ns["is_empty_text"]("") is True
    assert ns["is_empty_text"]("x") is False
    assert ns["capitalize"]("hello world") == "Hello world"
    assert ns["capitalize"]("") == ""
    assert ns["contains_any"]("hello", ["xyz", "ell"]) is True
    assert ns["contains_any"]("hello", ["xyz", "abc"]) is False
    assert ns["count_lines"]("a\nb\nc") == 3
    assert ns["count_lines"]("") == 0


def test_algo_added_helpers_python_backend(tmp_path: Path):
    ns = _extra_module_ns(tmp_path, "algo")

    assert ns["chunk_sum"]([1, 2, 3, 4, 5], 2) == [3, 7, 5]
    assert ns["chunk_sum"]([1, 2, 3], 0) == []
    assert ns["running_max"]([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]
    assert ns["running_max"]([]) == []

    index = ns["find_index"]([1, 2, 3], 2)
    assert index.is_some()
    assert index.unwrap() == 1
    assert ns["find_index"]([1, 2, 3], 9).is_none()

    assert ns["all_equal"]([2, 2, 2]) is True
    assert ns["all_equal"]([2, 2, 3]) is False
    assert ns["all_equal"]([]) is True
    assert ns["rotate_left"]([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert ns["rotate_left"]([1, 2, 3, 4], 7) == [4, 1, 2, 3]
    assert ns["zip_sum"]([1, 2, 3], [10, 20]) == [11, 22]
    assert ns["zip_sum"]([], [1]) == []


def test_prelude_exposes_added_pure_helpers(tmp_path: Path):
    ns = _extra_module_ns(tmp_path, "prelude")
    assert ns["capitalize"]("hi") == "Hi"
    assert ns["zip_sum"]([1, 2], [3, 4]) == [4, 6]
    assert "read_text_or" in ns


@pytest.mark.native
def test_str_utils_added_helpers_native_backend(native_ns):
    assert native_ns["trim_matches"]("xxabcxx", "xy") == "abc"
    assert native_ns["replace_first"]("a-b-c", "-", "+") == "a+b-c"
    assert native_ns["split_once"]("a:b:c", ":") == "b:c"
    assert native_ns["split_once"]("abc", ":") is None
    assert native_ns["is_empty_text"]("") is True
    assert native_ns["capitalize"]("hello world") == "Hello world"
    assert native_ns["contains_any"]("hello", ["xyz", "ell"]) is True
    assert native_ns["count_lines"]("a\nb\nc") == 3


@pytest.mark.native
def test_algo_added_helpers_native_backend(native_ns):
    assert native_ns["chunk_sum"]([1, 2, 3, 4, 5], 2) == [3, 7, 5]
    assert native_ns["chunk_sum"]([1, 2, 3], 0) == []
    assert native_ns["running_max"]([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]
    assert native_ns["find_index"]([1, 2, 3], 2) == 1
    assert native_ns["find_index"]([1, 2, 3], 9) is None
    assert native_ns["all_equal"]([2, 2, 2]) is True
    assert native_ns["rotate_left"]([1, 2, 3, 4], 7) == [4, 1, 2, 3]
    assert native_ns["zip_sum"]([1, 2, 3], [10, 20]) == [11, 22]
