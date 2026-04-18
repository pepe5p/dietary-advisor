"""Curated list of clinical-guideline source documents for the RAG corpus.

Most of these PDFs are publicly hosted by their issuing body. URLs occasionally
shift, so the ingest script tolerates 404s for individual entries and reports
the failures at the end of the run. A small set of bundled "seed" excerpts is
shipped in `seeds/` so the system has *some* knowledge even before the user
runs `ingest-corpus`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusSource:
    doc_id: str
    title: str
    url: str | None  # None means "seed-only" (no remote fetch)
    section: str = "main"
    tags: tuple[str, ...] = field(default_factory=tuple)


# Public PDFs, prioritising stable government / NGO endpoints. URLs are
# best-effort; if any 404 the user is told to drop the file manually into
# `dietary_advisor/knowledge/corpus/`.
SOURCES: list[CorpusSource] = [
    CorpusSource(
        doc_id="WHO_HEALTHY_DIET",
        title="WHO - Healthy diet (fact sheet)",
        url="https://cdn.who.int/media/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
        tags=("who", "general"),
    ),
    CorpusSource(
        doc_id="WHO_SODIUM_2023",
        title="WHO global report on sodium intake reduction (2023)",
        url="https://iris.who.int/bitstream/handle/10665/366393/9789240069985-eng.pdf",
        tags=("who", "sodium", "hypertension"),
    ),
    CorpusSource(
        doc_id="USDA_DGA_2020_2025",
        title="Dietary Guidelines for Americans 2020-2025",
        url="https://www.dietaryguidelines.gov/sites/default/files/2021-03/Dietary_Guidelines_for_Americans-2020-2025.pdf",
        tags=("usda", "general"),
    ),
    CorpusSource(
        doc_id="ADA_NUTRITION_2019",
        title="ADA: Nutrition Therapy for Adults With Diabetes or Prediabetes (2019)",
        url="https://diabetesjournals.org/care/article-pdf/42/5/731/553088/dci190014.pdf",
        tags=("ada", "diabetes"),
    ),
    CorpusSource(
        doc_id="NICE_NG28",
        title="NICE NG28: Type 2 diabetes in adults: management",
        url="https://www.nice.org.uk/guidance/ng28/resources/type-2-diabetes-in-adults-management-pdf-1837338615493",
        tags=("nice", "diabetes"),
    ),
    CorpusSource(
        doc_id="NICE_NG136",
        title="NICE NG136: Hypertension in adults: diagnosis and management",
        url="https://www.nice.org.uk/guidance/ng136/resources/hypertension-in-adults-diagnosis-and-management-pdf-66141722710213",
        tags=("nice", "hypertension"),
    ),
    CorpusSource(
        doc_id="EFSA_DRV_SUMMARY",
        title="EFSA Dietary Reference Values - summary report",
        url="https://www.efsa.europa.eu/sites/default/files/2017_09_DRVs_summary_report.pdf",
        tags=("efsa", "drv"),
    ),
]
