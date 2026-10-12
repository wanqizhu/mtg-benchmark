"""Render an offline Plotly dashboard. All observations are embedded locally."""
import json
from pathlib import Path

from plotly.offline import get_plotlyjs
import markdown


def render(data, destination):
    template = (Path(__file__).parent / "dashboard_template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    destination.write_text(template.replace("__PLOTLY_LIBRARY__", get_plotlyjs()).replace("__EVALUATION_DATA__", payload))
    style = template.split('<style>')[1].split('</style>')[0]
    report = markdown.markdown((destination.parent / 'report.md').read_text(), extensions=['tables', 'fenced_code'])
    report = report.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
    (destination.parent / 'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FRA evaluation report</title><style>' + style + 'pre{overflow-x:auto;padding:16px;background:var(--surface)}code{font-size:13px}h3{margin-top:24px}</style></head><body><main><nav><a href="dashboard.html">Interactive plots</a><a href="predictions_long.csv">Card/source data</a><a href="report.md">Markdown report</a></nav>' + report + '</main></body></html>')


if __name__ == "__main__":
    root = Path(__file__).parent
    render(json.loads((root / "results/dashboard_data.json").read_text()), root / "results/dashboard.html")
