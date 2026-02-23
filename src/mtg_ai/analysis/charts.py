from __future__ import annotations


def ascii_line_chart(points: list[tuple[int, float]], width: int = 64, height: int = 12) -> str:
    if not points:
        return "(no data)"
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in points:
        col = int((x - min_x) / (max_x - min_x) * (width - 1))
        row = int((y - min_y) / (max_y - min_y) * (height - 1))
        row = (height - 1) - row
        canvas[row][col] = "*"
    lines = ["".join(row) for row in canvas]
    lines.append(f"x:[{min_x}, {max_x}] y:[{min_y:.3f}, {max_y:.3f}]")
    return "\n".join(lines)


def bar_chart(items: list[tuple[str, float]], width: int = 48) -> str:
    if not items:
        return "(no data)"
    max_value = max(value for _, value in items)
    if max_value <= 0:
        max_value = 1.0
    lines: list[str] = []
    for label, value in items:
        bar_len = int((value / max_value) * width)
        lines.append(f"{label:24} {'#' * bar_len} {value:.3f}")
    return "\n".join(lines)

