"""Curated list of clinical-guideline source documents for the RAG corpus.

Most of these PDFs are publicly hosted by their issuing body. URLs occasionally
shift, so the ingest script tolerates 404s for individual entries and reports
the failures at the end of the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusSource:
    doc_id: str
    title: str
    url: str


# Public PDFs, prioritising stable government / NGO endpoints. URLs are
# best-effort; if any 404 the user is told to drop the file manually into
# `.data/rag/corpus/`.
SOURCES: list[CorpusSource] = [
    CorpusSource(
        doc_id="WHO_HEALTHY_DIET",
        title="WHO - Healthy diet (fact sheet)",
        url="https://cdn.who.int/media/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
    ),
    CorpusSource(
        doc_id="NICE_NG28",
        title="NICE NG28: Type 2 diabetes in adults: management",
        url="https://www.nice.org.uk/guidance/ng28/resources/type-2-diabetes-in-adults-management-pdf-1837338615493",
    ),
    CorpusSource(
        doc_id="NICE_NG136",
        title="NICE NG136: Hypertension in adults: diagnosis and management",
        url="https://www.nice.org.uk/guidance/ng136/resources/hypertension-in-adults-diagnosis-and-management-pdf-66141722710213",
    ),
    CorpusSource(
        doc_id="EFSA_DRV_SUMMARY",
        title="EFSA Dietary Reference Values - summary report",
        url="https://www.efsa.europa.eu/sites/default/files/2017_09_DRVs_summary_report.pdf",
    ),
]
