"""Stable dependency ordering for struct declarations."""

from ast_nodes import StructDecl, TypeName


def order_struct_declarations(
    structs: list[StructDecl],
) -> list[StructDecl]:
    by_name = {struct.name: struct for struct in structs}
    ordered: list[StructDecl] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(struct: StructDecl) -> None:
        if struct.name in visited or struct.name in visiting:
            return
        visiting.add(struct.name)
        for _field_name, field_type, _is_public in struct.fields:
            if isinstance(field_type, TypeName):
                dependency = by_name.get(field_type.name)
                if dependency is not None:
                    visit(dependency)
        visiting.remove(struct.name)
        visited.add(struct.name)
        ordered.append(struct)

    for struct in structs:
        visit(struct)
    return ordered
