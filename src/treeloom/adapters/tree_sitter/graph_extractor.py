from pathlib import Path

from tree_sitter_language_pack import get_language
from tree_sitter import Parser as TSParser, Language, Query, QueryCursor

from treeloom.adapters.tree_sitter.indexer import LANGUAGE_MAP


def _run_query(lang, query_str, root_node):
    q = Query(lang, query_str)
    cursor = QueryCursor(q)
    matches = cursor.matches(root_node)
    cap_groups: dict[str, list] = {}
    for _pattern_idx, captures_dict in matches:
        for cap_name, nodes in captures_dict.items():
            cap_groups.setdefault(cap_name, []).extend(nodes)
    return cap_groups

Entity = dict
Relationship = dict

LANGUAGE_QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "class": """
            (class_definition
                name: (identifier) @class.name
                (argument_list (identifier) @class.base)*
            ) @class.def
        """,
        "function": """
            (function_definition
                name: (identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_statement (dotted_name) @import.module) @import.stmt
            (import_from_statement
                module_name: (dotted_name) @import.module
                (dotted_name) @import.name
            ) @import.stmt
        """,
        "call": """
            (call function: (identifier) @call.name) @call.expr
        """,
    },
    "javascript": {
        "class": """
            (class_declaration
                name: (identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_declaration
                name: (identifier) @function.name
            ) @function.def
            (method_definition
                name: (property_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_statement
                source: (string) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "typescript": {
        "class": """
            (class_declaration
                name: (identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_declaration
                name: (identifier) @function.name
            ) @function.def
            (method_definition
                name: (property_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_statement
                source: (string) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "go": {
        "class": "",
        "function": """
            (function_declaration
                name: (identifier) @function.name
            ) @function.def
            (method_declaration
                name: (field_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_declaration
                (import_spec_list
                    (import_spec path: (interpreted_string_literal) @import.module))
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "java": {
        "class": """
            (class_declaration
                name: (identifier) @class.name
                (superclass (type_identifier) @class.base)?
            ) @class.def
        """,
        "function": """
            (method_declaration
                name: (identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_declaration
                (scoped_identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (method_invocation
                name: (identifier) @call.name
            ) @call.expr
        """,
    },
    # NB: the key must match LANGUAGE_MAP's value ("csharp"), not tree-sitter's
    # historical grammar name "c_sharp" — _detect_language looks up by the
    # former. A "c_sharp" key sits dead while every .cs file silently falls
    # back to a Module-only entity (no headers, no graph signals, no
    # find_definition), which is exactly what happened to featbit (54% C#).
    "csharp": {
        "class": """
            (class_declaration
                name: (identifier) @class.name
            ) @class.def
            (interface_declaration
                name: (identifier) @class.name
            ) @class.def
            (struct_declaration
                name: (identifier) @class.name
            ) @class.def
            (record_declaration
                name: (identifier) @class.name
            ) @class.def
            (class_declaration
                (base_list (identifier) @class.base)
            )
        """,
        "function": """
            (method_declaration
                name: (identifier) @function.name
            ) @function.def
            (constructor_declaration
                name: (identifier) @function.name
            ) @function.def
            (local_function_statement
                name: (identifier) @function.name
            ) @function.def
        """,
        "import": """
            (using_directive
                (qualified_name) @import.module
            ) @import.stmt
            (using_directive
                (identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (invocation_expression
                function: (identifier) @call.name
            ) @call.expr
            (invocation_expression
                function: (member_access_expression
                    name: (identifier) @call.name
                )
            ) @call.expr
        """,
    },
    "tsx": {
        "class": """
            (class_declaration
                name: (identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_declaration
                name: (identifier) @function.name
            ) @function.def
            (method_definition
                name: (property_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_statement
                source: (string) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "rust": {
        # extract_graph only consumes the "class" and "function" keys; this
        # was keyed "struct" and never ran.
        "class": """
            (struct_item name: (type_identifier) @class.name) @class.def
        """,
        "function": """
            (function_item name: (identifier) @function.name) @function.def
        """,
        "import": """
            (use_declaration
                (scoped_identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "ruby": {
        "class": """
            (class
                name: (constant) @class.name
            ) @class.def
            (module
                name: (constant) @class.name
            ) @class.def
        """,
        "function": """
            (method
                name: (identifier) @function.name
            ) @function.def
            (singleton_method
                name: (identifier) @function.name
            ) @function.def
        """,
        "import": """
            (call
                method: (identifier) @_m
                arguments: (argument_list (string) @import.module)
                (#match? @_m "^(require|require_relative|load)$")
            ) @import.stmt
        """,
        "call": """
            (call
                method: (identifier) @call.name
            ) @call.expr
        """,
    },
    "php": {
        "class": """
            (class_declaration
                name: (name) @class.name
            ) @class.def
            (interface_declaration
                name: (name) @class.name
            ) @class.def
            (trait_declaration
                name: (name) @class.name
            ) @class.def
        """,
        "function": """
            (function_definition
                name: (name) @function.name
            ) @function.def
            (method_declaration
                name: (name) @function.name
            ) @function.def
        """,
        "import": """
            (namespace_use_clause
                (qualified_name) @import.module
            ) @import.stmt
        """,
        "call": """
            (function_call_expression
                function: (name) @call.name
            ) @call.expr
            (method_call_expression
                name: (name) @call.name
            ) @call.expr
        """,
    },
    "kotlin": {
        "class": """
            (class_declaration
                (type_identifier) @class.name
            ) @class.def
            (object_declaration
                (type_identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_declaration
                (simple_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_header
                (identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                (simple_identifier) @call.name
            ) @call.expr
        """,
    },
    "scala": {
        "class": """
            (class_definition
                name: (identifier) @class.name
            ) @class.def
            (object_definition
                name: (identifier) @class.name
            ) @class.def
            (trait_definition
                name: (identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_definition
                name: (identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_declaration
                (stable_identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "swift": {
        # swift's grammar parses `struct` as class_declaration too; names have
        # no `name:` field label.
        "class": """
            (class_declaration
                (type_identifier) @class.name
                (inheritance_specifier (user_type) @class.base)*
            ) @class.def
        """,
        "function": """
            (function_declaration
                (simple_identifier) @function.name
            ) @function.def
        """,
        "import": """
            (import_declaration
                (identifier) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                (simple_identifier) @call.name
            ) @call.expr
        """,
    },
    "c": {
        "class": """
            (struct_specifier
                name: (type_identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @function.name
                )
            ) @function.def
        """,
        "import": """
            (preproc_include
                path: (_) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
    "cpp": {
        "class": """
            (class_specifier
                name: (type_identifier) @class.name
            ) @class.def
            (struct_specifier
                name: (type_identifier) @class.name
            ) @class.def
        """,
        "function": """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @function.name
                )
            ) @function.def
            (function_definition
                declarator: (function_declarator
                    declarator: (qualified_identifier) @function.name
                )
            ) @function.def
        """,
        "import": """
            (preproc_include
                path: (_) @import.module
            ) @import.stmt
        """,
        "call": """
            (call_expression
                function: (identifier) @call.name
            ) @call.expr
        """,
    },
}


def extract_graph(file_path: str, source: str) -> tuple[list[Entity], list[Relationship]]:
    lang = _detect_language(file_path)
    if not lang or lang not in LANGUAGE_QUERIES:
        return _fallback_module_entity(file_path, source)

    queries = LANGUAGE_QUERIES[lang]
    try:
        lang_obj = get_language(lang)
    except Exception:
        return _fallback_module_entity(file_path, source)

    try:
        parser = TSParser()
        parser.language = lang_obj
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return _fallback_module_entity(file_path, source)

    source_bytes = source.encode("utf-8")
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    module_id = f"module://{file_path}"

    entities.append({
        "id": module_id,
        "type": "Module",
        "name": file_path,
        "file_path": file_path,
        "start_line": 1,
        "end_line": source.count("\n") + 1,
        "language": lang,
    })

    class_entities: dict[str, Entity] = {}
    func_entities: dict[str, Entity] = {}
    class_parents: dict[str, list[str]] = {}

    for entity_type in ("class", "function"):
        qs = queries.get(entity_type, "").strip()
        if not qs:
            continue
        try:
            cap_groups = _run_query(lang_obj, qs, tree.root_node)
        except Exception:
            continue

        def_nodes = cap_groups.get(f"{entity_type}.def", [])
        name_nodes = cap_groups.get(f"{entity_type}.name", [])

        if entity_type == "class":
            base_nodes = cap_groups.get(f"{entity_type}.base", [])
            base_by_def: dict[int, list[str]] = {}
            for bn in base_nodes:
                for i, dn in enumerate(def_nodes):
                    if dn.start_byte <= bn.start_byte < dn.end_byte:
                        base_by_def.setdefault(i, []).append(
                            source_bytes[bn.start_byte:bn.end_byte].decode()
                        )
                        break

            for i, class_node in enumerate(def_nodes):
                if i >= len(name_nodes):
                    continue
                name = source_bytes[name_nodes[i].start_byte:name_nodes[i].end_byte].decode()
                start_line = class_node.start_point[0] + 1
                end_line = class_node.end_point[0] + 1
                bases = base_by_def.get(i, [])
                sig = f"class {name}({', '.join(bases)})" if bases else f"class {name}"
                ent: Entity = {
                    "id": f"class://{file_path}#{name}",
                    "type": "Class",
                    "name": name,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": sig,
                    "language": lang,
                }
                entities.append(ent)
                class_entities[name] = ent
                class_parents[name] = bases
                relationships.append({
                    "source_id": module_id,
                    "target_id": ent["id"],
                    "type": "DEFINES",
                })
        elif entity_type == "function":
            for i, func_node in enumerate(def_nodes):
                if i >= len(name_nodes):
                    continue
                name = source_bytes[name_nodes[i].start_byte:name_nodes[i].end_byte].decode()
                start_line = func_node.start_point[0] + 1
                end_line = func_node.end_point[0] + 1
                # errors="replace": the fixed 80-byte cap can split a
                # multibyte UTF-8 char (e.g. an em-dash in a docstring) —
                # strict decoding threw and cost the whole FILE its graph.
                # File-level encoding fallback is handled by
                # infrastructure.fileio.read_source; the replace here only
                # guards the 80-byte cap splitting a multibyte char.
                sig = source_bytes[func_node.start_byte:min(func_node.start_byte + 80, func_node.end_byte)].decode(errors="replace")
                ent: Entity = {
                    "id": f"func://{file_path}#{name}",
                    "type": "Function",
                    "name": name,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": sig,
                    "language": lang,
                }
                entities.append(ent)
                func_entities[name] = ent
                relationships.append({
                    "source_id": module_id,
                    "target_id": ent["id"],
                    "type": "DEFINES",
                })

    import_qs = queries.get("import", "").strip()
    if import_qs:
        try:
            igroups = _run_query(lang_obj, import_qs, tree.root_node)
            module_caps = igroups.get("import.module", [])
            for mc in module_caps:
                mod_name = source_bytes[mc.start_byte:mc.end_byte].decode()
                target_id = f"module://{_resolve_import_path(mod_name, file_path)}"
                relationships.append({
                    "source_id": module_id,
                    "target_id": target_id,
                    "type": "IMPORTS",
                })
        except Exception:
            pass

    call_qs = queries.get("call", "").strip()
    if call_qs and func_entities:
        try:
            cgroups = _run_query(lang_obj, call_qs, tree.root_node)
            call_names = cgroups.get("call.name", [])
            for cn in call_names:
                name = source_bytes[cn.start_byte:cn.end_byte].decode()
                caller_func = _find_enclosing_func(cn, func_entities.values())
                if caller_func and name in func_entities:
                    relationships.append({
                        "source_id": caller_func["id"],
                        "target_id": func_entities[name]["id"],
                        "type": "CALLS",
                    })
        except Exception:
            pass

    for class_name, bases in class_parents.items():
        for base in bases:
            base_id = f"class://{file_path}#{base}"
            relationships.append({
                "source_id": class_entities[class_name]["id"],
                "target_id": base_id,
                "type": "INHERITS",
            })

    return entities, relationships


def _detect_language(file_path: str) -> str | None:
    ext = Path(file_path).suffix
    return LANGUAGE_MAP.get(ext)


def _resolve_import_path(module_name: str, current_file: str) -> str:
    module_name = module_name.strip("\"'")
    ext = Path(current_file).suffix
    suffix_map = {
        ".cs": ".cs", ".ts": ".ts", ".tsx": ".ts", ".js": ".ts", ".jsx": ".ts",
        ".go": ".go", ".rs": ".rs", ".java": ".java",
        ".rb": ".rb", ".php": ".php", ".kt": ".kt", ".scala": ".scala",
        ".c": ".c", ".h": ".h", ".cpp": ".cpp", ".hpp": ".hpp",
    }
    suffix = suffix_map.get(ext, ".py")
    if module_name.startswith("./") or module_name.startswith("../"):
        cur_dir = Path(current_file).parent
        resolved = (cur_dir / module_name).resolve()
        if not resolved.suffix:
            resolved = resolved.with_suffix(suffix)
        return str(resolved)
    if "/" in module_name:
        return module_name
    rel = Path(module_name.replace("::", "/").replace(".", "/"))
    return str(rel.with_suffix(suffix))


def _find_enclosing_func(node, func_entities: list[Entity]) -> Entity | None:
    node_start = node.start_byte
    for fe in sorted(func_entities, key=lambda x: -x.get("end_line", 0)):
        fe_start = fe.get("start_line", 0)
        fe_end = fe.get("end_line", 0)
        if fe_start <= node.start_point[0] + 1 <= fe_end:
            return fe
    return None


def _fallback_module_entity(file_path: str, source: str) -> tuple[list[Entity], list[Relationship]]:
    return [
        {
            "id": f"module://{file_path}",
            "type": "Module",
            "name": file_path,
            "file_path": file_path,
            "start_line": 1,
            "end_line": source.count("\n") + 1,
            "language": _detect_language(file_path) or "",
        }
    ], []
