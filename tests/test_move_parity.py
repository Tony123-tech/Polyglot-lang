"""E009 parity: the Python and native backends must agree on ownership.

The Python backend has no ownership at runtime, so before this pass the same
program could run there and fail on the native backend with a rustc
"use of moved value". Every case below is checked twice: once through the
Python backend and once through the native backend's front end, which runs
before cargo is invoked.
"""

from __future__ import annotations

import pathlib

import pytest

from backends.move_check import MoveError, check_moves
from backends.python.translator import PythonBackend
from backends.rust.compiler import NativeBackend
from lexer import lex
from parser import parse

# (name, source, expected return value of main)
ALLOWED = [
    (
        "clone_then_two_uses",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("hi");
    let copy = text.clone();
    let a = take(copy);
    let b = take(text);
    a + b
}
''',
        4,
    ),
    (
        "borrow_in_print",
        '''
fn main() -> i64 {
    let text = String::from("hi");
    println!("{}", text);
    println!("{}", text);
    7
}
''',
        7,
    ),
    (
        "copy_scalar_reuse",
        '''
fn main() -> i64 {
    let n = 3;
    let m = n + n;
    let k = n * m;
    k
}
''',
        18,
    ),
    (
        "index_clone",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let words = vec![String::from("a"), String::from("bb")];
    let a = take(words[0].clone());
    let b = take(words[1].clone());
    a + b
}
''',
        3,
    ),
    (
        "early_return_guard",
        '''
fn sort_like(values: Vec<i64>) -> Vec<i64> {
    let mut out = values.clone();
    let n = out.len();
    if n < 2 { return out; }
    out[0] = out[1];
    out
}
fn main() -> i64 {
    let sorted = sort_like(vec![5, 3]);
    sorted.len() as i64
}
''',
        2,
    ),
    (
        "move_inside_loop_body",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let seed = String::from("ab");
    let mut i = 0;
    let mut total = 0;
    while i < 3 {
        let copy = seed.clone();
        total = total + take(copy);
        i = i + 1;
    }
    total
}
''',
        6,
    ),
    (
        "exclusive_branch_moves",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("abc");
    let mut n = 0;
    if n == 0 {
        n = take(text);
    } else {
        n = take(text);
    }
    n
}
''',
        3,
    ),
    (
        "shadowing_after_move",
        '''
fn main() -> i64 {
    let text = String::from("old");
    let copy = text;
    let text = String::from("new");
    println!("{}", text);
    5
}
''',
        5,
    ),
]

# (name, source, expected message substring, expected moved/use line pair)
REJECTED = [
    (
        "second_call",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("hi");
    take(text);
    take(text);
    0
}
''',
        "use of moved value: `text`",
        (5, 6),
    ),
    (
        "move_then_read",
        '''
fn main() -> i64 {
    let text = String::from("hi");
    let moved = text;
    println!("{}", text);
    0
}
''',
        "use of moved value: `text`",
        (4, 5),
    ),
    (
        "return_value_reused",
        '''
fn consume(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("abc");
    let one = consume(text);
    println!("{} {}", one, text);
    0
}
''',
        "use of moved value: `text`",
        (5, 6),
    ),
    (
        "vec_second_call",
        '''
fn take_all(values: Vec<i64>) -> i64 { values.len() as i64 }
fn main() -> i64 {
    let values = vec![1, 2, 3];
    take_all(values);
    take_all(values);
    0
}
''',
        "use of moved value: `values`",
        (5, 6),
    ),
    (
        "index_move_out",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let words = vec![String::from("a"), String::from("b")];
    take(words[0]);
    0
}
''',
        "cannot move out of index of `words`",
        (5, 5),
    ),
    (
        "if_branch_move",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("hi");
    let flag = 1;
    if flag > 0 { take(text); }
    println!("{}", text);
    0
}
''',
        "use of moved value: `text`",
        (6, 7),
    ),
    (
        "loop_body_move",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("hi");
    let mut i = 0;
    while i < 2 {
        take(text);
        i = i + 1;
    }
    println!("{}", text);
    0
}
''',
        "use of moved value: `text`",
        (7, 10),
    ),
    (
        "for_move_then_len",
        '''
fn main() -> i64 {
    let values = vec![1, 2, 3];
    for item in values {
        println!("{}", item);
    }
    values.len() as i64
}
''',
        "use of moved value: `values`",
        (4, 7),
    ),
    (
        "push_twice",
        '''
fn main() -> i64 {
    let name = String::from("a");
    let mut names = vec![String::from("z")];
    names.push(name);
    names.push(name);
    names.len() as i64
    }
''',
        "use of moved value: `name`",
        (5, 6),
    ),
    (
        "some_wrap_then_read",
        '''
fn main() -> i64 {
    let name = String::from("a");
    let wrapped = Some(name);
    println!("{}", name);
    0
}
''',
        "use of moved value: `name`",
        (4, 5),
    ),
    (
        "struct_field_move",
        '''
struct Point { label: String, x: i64 }
fn main() -> i64 {
    let name = String::from("origin");
    let point = Point { label: name, x: 0 };
    println!("{}", name);
    point.x
}
''',
        "use of moved value: `name`",
        (5, 6),
    ),
    (
        "match_arm_move",
        '''
fn take(s: String) -> i64 { s.len() as i64 }
fn main() -> i64 {
    let text = String::from("hi");
    let flag = 1;
    match flag {
        1 => { take(text); }
        _ => { }
    }
    println!("{}", text);
    0
}
''',
        "use of moved value: `text`",
        (7, 10),
    ),
]


def _program(source: str):
    return parse(lex(source))


@pytest.mark.parametrize("name,source,expected", ALLOWED, ids=[case[0] for case in ALLOWED])
def test_allowed_programs_run_on_python(name: str, source: str, expected: int) -> None:
    program = _program(source)
    namespace = PythonBackend(trusted=True).compile(program)
    assert namespace["main"]() == expected


@pytest.mark.parametrize("name,source,expected", ALLOWED, ids=[case[0] for case in ALLOWED])
def test_allowed_programs_pass_the_shared_check(name: str, source: str, expected: int) -> None:
    check_moves(_program(source))


@pytest.mark.parametrize(
    "name,source,message,lines", REJECTED, ids=[case[0] for case in REJECTED]
)
def test_rejected_programs_raise_identical_e009(
    name: str, source: str, message: str, lines: tuple[int, int]
) -> None:
    program = _program(source)

    with pytest.raises(MoveError) as python_error:
        PythonBackend(trusted=True).compile(program)

    with pytest.raises(MoveError) as native_error:
        NativeBackend(cache_dir=pathlib.Path("/tmp/polyglot-move-parity-cache")).build(program)

    assert str(python_error.value).splitlines()[0] == str(native_error.value).splitlines()[0]
    assert message in str(python_error.value)
    assert python_error.value.moved_at == lines[0]
    assert python_error.value.line == lines[1]


def test_e009_text_is_the_standard_five_lines() -> None:
    program = _program(REJECTED[0][1])
    with pytest.raises(MoveError) as error:
        check_moves(program)
    rendered = str(error.value)
    assert rendered.splitlines()[0] == "error[E009]: use of moved value: `text`"
    assert rendered.splitlines()[1].strip().startswith("reason:")
    assert rendered.splitlines()[2].strip().startswith("help:")
    assert rendered.splitlines()[3].strip().startswith("note:")
    assert rendered.splitlines()[4].strip().startswith("at: line")


@pytest.mark.slow
@pytest.mark.parametrize("name,source,expected", ALLOWED[:3], ids=[case[0] for case in ALLOWED[:3]])
def test_allowed_programs_build_and_run_natively(
    name: str, source: str, expected: int
) -> None:
    program = _program(source)
    namespace = NativeBackend(cache_dir=pathlib.Path("/tmp/polyglot-move-parity-native")).compile(
        program
    )
    assert namespace["main"]() == expected
