"""
Polyglot demo: Rust syntax, Python semantics.

Run:
    python3 main.py
"""


from backends.python.translator import compile_rust
from lexer import Lexer
from parser import Parser

SRC = r'''
fn add(a: i64, b: i64) -> i64
    a + b
}

fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    n * factorial(n - 1)
}

fn main() -> i64 {
    let x: i64 = 0xFF_FF;
    let y = 1_000_000 + 1.5e-3;
    let msg = "hello";
    println!("{} {} {}", msg, x, y);
    let total = 0;
    for i in 0..=10 {
        total = total + i;
    }
    println!("sum = {}", total);
    println!("add = {}", add(1, 2));
    println!("fact = {}", factorial(5));
    total
}
'''


def main():
    tokens = Lexer(SRC).lex()
    program = Parser(tokens).parse_program()
    ns = compile_rust(program)
    ns["main"]()
    print("add(1, 2) =", ns["add"](1, 2))
    print("add('a', 'b') =", ns["add"]("a", "b"))
    print("factorial(5) =", ns["factorial"](5))


if __name__ == "__main__":
    main()
