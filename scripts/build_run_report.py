"""Build a compact checkpoint reference from the committed run records."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from guardstudy.io import ROOT, read_json


def main():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name="Small", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=7)
    )
    styles["Title"].fontSize = 23
    styles["Title"].leading = 27
    out = ROOT / "reports/run-tracking.pdf"
    story = []

    def p(text, style="Small"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths, color="#e9eff8", font=8.5):
        t = Table(
            rows,
            colWidths=widths,
            rowHeights=[28] + [16] * (len(rows) - 1),
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), font),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(color)),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f6f8fa")],
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172435")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#acbacb")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend([t, Spacer(1, 14)])

    roster = read_json(ROOT / "data/runs/checkpoints.json")["checkpoints"]
    hist = {
        r["checkpoint_id"]: r
        for r in read_json(ROOT / "data/runs/training-history.json.gz")["records"]
    }
    p("Generalizable Safeguards", "Title")
    p("Final checkpoint and training reference", "Heading2")
    p(
        "This report identifies the checkpoints behind the current paper. It records selected updates and recovered validation history. The original campaigns and Overleaf document are unchanged."
    )
    table(
        [
            ["Model", "Trained checkpoints", "SFT baselines", "Adapter rank"],
            ["Qwen3.5-9B", "101", "1", "64"],
            ["GPT-OSS-120B", "24", "1", "32"],
        ],
        [180, 115, 100, 115],
    )
    p("Training recipe", "Heading2")
    table(
        [
            ["Setting", "Value"],
            ["Training domains", "Privacy, fraud, defamation"],
            ["Original training rows", "420 SafePyramid rows"],
            ["Appended supplement", "1,210 PolicyTraj rows"],
            ["Validation questions", "90"],
            ["Responses per prompt", "10"],
            ["Prompts per attempt", "3, one per training domain"],
            ["Thinking during training", "Off"],
            ["Validation cadence", "Every ten successful updates"],
            ["Final-answer token allowance", "23,000"],
            [
                "Thinking-on evaluation",
                "10,000 reasoning tokens, then 23,000 answer tokens",
            ],
        ],
        [190, 320],
    )
    p(
        "Qwen uses nine recorded training seeds. GPT uses the subset 2, 4, and 5. Each contrast lists its actual eligible paired seeds. Results compare reward choices within each model."
    )
    p("Evaluation coverage", "Heading2")
    table(
        [
            ["Panel", "Questions", "Answers per question", "Thinking"],
            ["Out-of-distribution domains", "926", "3", "On and off"],
            ["In-distribution domains", "400", "1", "On"],
            ["Fixed conversation, policy levels", "84", "1", "On"],
        ],
        [200, 75, 130, 105],
    )
    p(
        "The recovered validation history covers all 125 trained checkpoints through each reported update. Later historical training records are excluded from this reference."
    )
    story.append(PageBreak())
    for family, title, color in [
        ("Qwen", "Qwen3.5-9B", "#e7effb"),
        ("GPT", "GPT-OSS-120B", "#fff0de"),
    ]:
        p(title + " checkpoint roster", "Heading1")
        p(
            "The selected update identifies the evaluated weights. Validation entries count recovered records through that update. They do not count optimizer steps."
        )
        rows = [
            [
                "Checkpoint identifier",
                "Correctness",
                "Arm",
                "Seed",
                "Selected\nupdate",
                "Validation\nentries",
            ]
        ]
        for r in sorted(
            (r for r in roster if r["model_family"] == family and r["target"] != "SFT"),
            key=lambda r: (r["target"], r["arm"], r["seed"]),
        ):
            rows.append(
                [
                    r["checkpoint_id"],
                    {"jaccard": "Jaccard", "f1": "Set F1", "binary": "Exact set"}[r["target"]],
                    r["arm"],
                    str(r["seed"]),
                    str(r["update"]),
                    str(len(hist[r["checkpoint_id"]]["history"])),
                ]
            )
        table(rows, [215, 78, 35, 40, 70, 72], color, 8)
        p(
            "The full roster is reports/checkpoint-roster.csv. Metric histories are in reports/training-history.csv.gz. Original state and selection records are retained in the local archive."
        )
        if family == "Qwen":
            story.append(PageBreak())

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#607085"))
        canvas.drawString(42, 23, "Generalizable Safeguards | Local review package")
        canvas.drawRightString(570, 23, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=42,
        rightMargin=42,
        topMargin=36,
        bottomMargin=38,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
