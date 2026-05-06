from __future__ import annotations


class DefaultPromptStrategy:
    code_format_instruction = """
Return only valid Python 3 code.
Do not include markdown fences.
The program must read from stdin and print to stdout using the format from the task.
A simple fast heuristic is acceptable.
""".strip()

    gen_instruction = """
You are writing a Python program for an optimization problem.
Generate a complete solution from scratch.
""".strip()

    combine_instruction = """
You are given several candidate solutions.
Combine their best ideas into one improved Python solution.
""".strip()

    mutate_instruction = """
You are given a problem and one candidate solution.
Improve the solution while keeping it correct and reasonably fast.
""".strip()

    fix_instruction = """
You are given a problem and one candidate solution which is incorrect.
Fix the solution while keeping it correct and reasonably fast.
Return only valid Python 3 code.
""".strip()

    def _build_prompt(
        self,
        instruction: str,
        problem_text: str,
        extra_sections: list[tuple[str, str]] | None = None,
    ) -> str:
        parts = [
            "/no_think",
            "Return only valid Python 3 code.",
            "No explanations.",
            "Do not output triple backticks.",
            "Do not output markdown.",
            "If you cannot solve the task, still output a valid Python program.",
            "",
            "### Instruction",
            instruction,
            "",
            "### Output requirements",
            self.code_format_instruction,
            "",
            "### Task",
            problem_text.strip(),
        ]

        if extra_sections:
            for title, body in extra_sections:
                parts.extend(["", f"### {title}", body.strip()])

        return "\n".join(parts) + "\n"

    def build_gen_prompt(self, problem_text: str) -> str:
        return self._build_prompt(self.gen_instruction, problem_text)

    def build_combine_prompt(self, problem_text: str, candidate_codes: list[str]) -> str:
        extra_sections = [
            (f"Candidate solution {i + 1}", code)
            for i, code in enumerate(candidate_codes)
        ]
        return self._build_prompt(self.combine_instruction, problem_text, extra_sections)

    def build_mutate_prompt(self, problem_text: str, candidate_code: str) -> str:
        extra_sections = [("Candidate solution", candidate_code)]
        return self._build_prompt(self.mutate_instruction, problem_text, extra_sections)

    def build_fix_prompt(self, problem_text: str, candidate_code: str, issue: str) -> str:
        extra_sections = [
            ("Candidate solution", candidate_code),
            ("Issue", issue),
        ]
        return self._build_prompt(self.fix_instruction, problem_text, extra_sections)