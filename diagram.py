# diagram.py
import re

def methods_to_flowchart_dot(methods_text: str) -> str:
    """
    Heuristic extraction of steps from Methods text and return a DOT string (Graphviz) for a simple left->right flowchart.
    The heuristic:
    - Look for numbered lists (1., 2., etc.) or bullet lines
    - If not present, split into sentences and keep the first ~6 sentences as steps
    """
    lines = methods_text.splitlines()
    steps = []

    # First try: numbered lines
    for line in lines:
        if re.match(r'^\s*\d+[\.\)]\s+', line):
            steps.append(re.sub(r'^\s*\d+[\.\)]\s+', '', line).strip())

    # Second: bullet points
    if not steps:
        for line in lines:
            if re.match(r'^\s*[-\u2022*]\s+', line):
                steps.append(re.sub(r'^\s*[-\u2022*]\s+', '', line).strip())

    # Third: sentences
    if not steps:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', methods_text.strip())
        # take top 6 non-trivial sentences
        for s in sentences:
            s = s.strip()
            if len(s) > 30:
                steps.append(s)
            if len(steps) >= 6:
                break

    # Fallback: short heuristics
    if not steps:
        steps = ["Read paper", "Extract data", "Preprocess", "Train model", "Evaluate", "Report results"]

    # Clean steps to reasonable length
    cleaned = []
    for s in steps:
        s = re.sub(r'\s+', ' ', s)
        if len(s) > 120:
            s = s[:117].rstrip() + "..."
        cleaned.append(s)

    # Build DOT left->right
    dot_lines = ["digraph G {", 'rankdir=LR;', 'node [shape=box, style=rounded];']
    for i, s in enumerate(cleaned):
        node_name = f"step{i+1}"
        label = s.replace('"', "'")
        dot_lines.append(f'{node_name} [label="{label}"];')
    for i in range(len(cleaned)-1):
        dot_lines.append(f'step{i+1} -> step{i+2};')
    dot_lines.append("}")
    return "\n".join(dot_lines)
