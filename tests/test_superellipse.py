"""Test approximated superellipse.

Note: This was mostly built using ChatGPT/Codex 5.6-Sol-high
"""

import pathlib

from geom2d.superellipse import rhombus_superellipse

def _svg_path_from_curves(curves: list[tuple]) -> str:
    """Convert the four cubic Bezier curves into one closed SVG path."""
    start = curves[0][0]
    parts = [f"M {start[0]:.3f},{start[1]:.3f}"]

    for _, c1, c2, end in curves:
        parts.append(
            f"C "
            f"{c1[0]:.3f},{c1[1]:.3f} "
            f"{c2[0]:.3f},{c2[1]:.3f} "
            f"{end[0]:.3f},{end[1]:.3f}"
        )

    parts.append("Z")

    return " ".join(parts)


def _write_test_svg(filename: str = "test3.svg") -> None:
    # A deliberately skewed/rotated rhombus.
    base_rhombus = [(30, 20), (220, 65), (250, 225), (60, 180)]

    # Negative, zero, circular, and increasingly square-ish cases.
    k_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    cell_width = 300
    cell_height = 300

    width = cell_width * 3
    rows = (len(k_values) + 2) // 3
    height = cell_height * rows

    svg = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )
    ]

    svg.append("""
    <style>
        text {
            font-family: sans-serif;
            font-size: 16px;
        }

        .rhombus {
            fill: none;
            stroke: #2bb673;
            stroke-width: 2;
        }

        .curve {
            fill: none;
            stroke: #087bea;
            stroke-width: 3;
        }

        .midpoint {
            fill: red;
        }

        .control-line {
            stroke: #999;
            stroke-width: 1;
            stroke-dasharray: 4 4;
        }

        .control-point {
            fill: #666;
        }
    </style>
    """)

    for index, k in enumerate(k_values):
        col = index % 3
        row = index // 3

        ox = col * cell_width + 20
        oy = row * cell_height + 45

        # translate rhombus
        rhombus = [(x + ox, y + oy) for x, y in base_rhombus]

        curves = rhombus_superellipse(rhombus, k)

        path = _svg_path_from_curves(curves)

        # Label
        n = 2.0**k

        svg.append(
            f'<text x="{ox + 5}" y="{oy - 15}">'
            f"K={k:g}, n={2.0**k:.4g}, curves={len(curves)}"
            f"</text>\n"
            f'<text x="{ox + 5}" y="{oy - 15}">'
            f"K={k:g}, n={n:.4g}"
            f"</text>"
        )

        # Draw rhombus and Bezier curve.
        point_string = " ".join(f"{x:.3f},{y:.3f}" for x, y in rhombus)

        svg.append(
            f'<polygon class="rhombus" points="{point_string}" />\n'
            f'<path class="curve" d="{path}" />'
        )

        # Draw midpoint and control-point diagnostics.
        #        for start, c1, c2, end in curves:
        #            svg.append(
        #                f'<line class="control-line" '
        #                f'x1="{start[0]}" y1="{start[1]}" '
        #                f'x2="{c1[0]}" y2="{c1[1]}" />'
        #            )
        #
        #            svg.append(
        #                f'<line class="control-line" '
        #                f'x1="{end[0]}" y1="{end[1]}" '
        #                f'x2="{c2[0]}" y2="{c2[1]}" />'
        #            )
        #
        #            svg.append(
        #                f'<circle class="control-point" '
        #                f'cx="{c1[0]}" cy="{c1[1]}" r="3" />'
        #            )
        #
        #            svg.append(
        #                f'<circle class="control-point" '
        #                f'cx="{c2[0]}" cy="{c2[1]}" r="3" />'
        #            )

        # Four edge midpoints.
        # mids = [
        #    (
        #        0.5 * (rhombus[i][0] + rhombus[(i + 1) % 4][0]),
        #        0.5 * (rhombus[i][1] + rhombus[(i + 1) % 4][1]),
        #    )
        #    for i in range(4)
        # ]

        # for x, y in mids:
        #    svg.append(
        #        f'<circle class="midpoint" '
        #        f'cx="{x}" cy="{y}" r="4" />'
        #    )

    svg.append("</svg>")

    pathlib.Path(filename).write_text("\n".join(svg), encoding="utf-8")

    print(f"Wrote {filename}")  # ruff: ignore[print]


if __name__ == "__main__":
    _write_test_svg()
