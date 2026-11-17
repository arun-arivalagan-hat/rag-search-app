"""Generate data/sample_docs/space_black_holes.pdf.

The PDF is the one sample document that is not plain text, so the ingestion
module (Module 1) has a real PDF to parse. The content lives here, in source
control, so the PDF can be regenerated deterministically:

    python scripts/generate_sample_pdf.py

Requires the ``dev`` extra (reportlab).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_docs" / "space_black_holes.pdf"

TITLE = "Black Holes"

PAGE_1 = [
    "A black hole is a region of spacetime where gravity is so strong that nothing, "
    "not even light, can escape once it crosses the boundary known as the event horizon. "
    "Black holes form when massive stars exhaust their nuclear fuel and collapse under "
    "their own gravity. Stars above roughly twenty times the mass of the Sun may leave "
    "behind a stellar-mass black hole after a supernova explosion.",
    "The size of the event horizon is described by the Schwarzschild radius, which grows "
    "in direct proportion to the mass of the black hole. For an object with the mass of "
    "the Sun, the Schwarzschild radius is only about three kilometres.",
    "Black holes cannot be seen directly because they emit no light. Astronomers detect "
    "them through their effects on surrounding matter: gas falling toward a black hole "
    "forms a hot accretion disk that glows brightly in X-rays, and stars orbiting an "
    "unseen companion reveal its mass through their motion.",
]

PAGE_2 = [
    "In 2019 the Event Horizon Telescope collaboration published the first image of a "
    "black hole's shadow, in the galaxy Messier 87. The result was produced by linking "
    "radio telescopes around the world into an Earth-sized virtual dish, a technique "
    "called very long baseline interferometry. In 2022 the same team imaged Sagittarius "
    "A*, the supermassive black hole at the centre of the Milky Way, which has a mass of "
    "about four million Suns.",
    "Gravitational waves provide a second window. In 2015 the LIGO detectors recorded "
    "ripples in spacetime from two black holes, each about thirty solar masses, spiralling "
    "together and merging roughly 1.3 billion light-years away. It was the first direct "
    "detection of gravitational waves, confirming a century-old prediction of Einstein's "
    "general theory of relativity.",
    "Stephen Hawking showed in 1974 that black holes should emit a faint thermal "
    "radiation, now called Hawking radiation, and therefore slowly lose mass over "
    "extremely long timescales.",
]


def build() -> Path:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, title=TITLE, author="RAG Lab sample corpus"
    )
    story = [Paragraph(TITLE, styles["Title"]), Spacer(1, 12)]
    for para in PAGE_1:
        story += [Paragraph(para, styles["BodyText"]), Spacer(1, 8)]
    story.append(PageBreak())
    for para in PAGE_2:
        story += [Paragraph(para, styles["BodyText"]), Spacer(1, 8)]
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
