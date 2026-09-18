"""Snippet import-stripping (sub-change 3).

`_strip_import_lines` drops self-contained import/using/#include directive lines
from a snippet body (that signal lives on the IMPORTS graph edges). It must NOT:
- touch a C# `using (...)` resource statement (real code),
- strip JS `export` (real code),
- partially strip a MULTI-line import (would leave dangling members),
- match a non-import line that merely starts with an import-ish token.
"""
from __future__ import annotations

from treeloom.application.retrieval import _strip_import_lines


def _lines(s: str) -> list[str]:
    return s.split("\n")


class TestStripsSingleLineImports:
    def test_python(self):
        out = _strip_import_lines("import os\nfrom x.y import z\n\ndef f():\n    return 1", "python")
        assert "import os" not in out and "from x.y import z" not in out
        assert "def f():" in out and "    return 1" in out

    def test_java_import_and_package(self):
        out = _strip_import_lines("package com.x;\nimport java.util.List;\nclass A {}", "java")
        assert "package" not in out and "import java.util.List;" not in out
        assert "class A {}" in out

    def test_csharp_using_directive(self):
        out = _strip_import_lines("using System;\nusing System.Linq;\npublic class C {}", "csharp")
        assert "using System;" not in out and "using System.Linq;" not in out
        assert "public class C {}" in out

    def test_javascript_import_and_require(self):
        src = "import y from 'z';\nimport { a } from 'x';\nconst fs = require('fs');\nconst c = 1;"
        out = _strip_import_lines(src, "javascript")
        assert "import y from 'z';" not in out
        assert "import { a } from 'x';" not in out
        assert "require('fs')" not in out
        assert "const c = 1;" in out

    def test_cpp_include(self):
        out = _strip_import_lines("#include <vector>\n#include \"foo.h\"\nint main(){}", "cpp")
        assert "#include" not in out and "int main(){}" in out

    def test_rust_use(self):
        out = _strip_import_lines("use std::collections::HashMap;\nfn main() {}", "rust")
        assert "use std" not in out and "fn main() {}" in out


class TestDoesNotStripRealCode:
    def test_csharp_using_statement_kept(self):
        # `using (var x = ...)` is a resource statement, not a directive.
        src = "using (var x = Open())\n{\n    x.Do();\n}"
        out = _strip_import_lines(src, "csharp")
        assert "using (var x = Open())" in out

    def test_js_export_kept(self):
        src = "export function f() {}\nexport default Foo;"
        out = _strip_import_lines(src, "javascript")
        assert "export function f() {}" in out and "export default Foo;" in out

    def test_python_identifier_starting_with_import_kept(self):
        # `import_data(...)` has no whitespace after `import` -> not a directive.
        out = _strip_import_lines("result = import_data()\nx = fromage", "python")
        assert "result = import_data()" in out and "x = fromage" in out

    def test_unknown_language_passthrough(self):
        src = "import os\ncode"
        assert _strip_import_lines(src, "yaml") == src
        assert _strip_import_lines(src, None) == src


class TestMultiLineImportsKeptWhole:
    def test_python_parenthesized_import_kept(self):
        # Opener `(` -> keep the whole multi-line import rather than strip the
        # head and orphan the members.
        src = "from x import (\n    a,\n    b,\n)\ncode = 1"
        out = _strip_import_lines(src, "python")
        assert "from x import (" in out and "    a," in out and ")" in out

    def test_js_destructured_multiline_kept(self):
        src = "import {\n  a,\n  b,\n} from 'x';\ncode"
        out = _strip_import_lines(src, "javascript")
        assert "import {" in out and "  a," in out


class TestHeaderPreserved:
    def test_chunk_header_comment_not_stripped(self):
        # The _chunk_header line is prepended as a comment; it must survive even
        # for a language whose directive keyword could look similar.
        src = "// in Class C · defines Foo\nusing System;\nvar x = 1;"
        out = _strip_import_lines(src, "csharp")
        assert "// in Class C · defines Foo" in out
        assert "using System;" not in out
        assert "var x = 1;" in out
