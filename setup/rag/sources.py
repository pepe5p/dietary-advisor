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
    # For auto-downloaded sources, the PDF endpoint to fetch. For `manual`
    # sources, the reference address a human should fetch the file from.
    url: str
    # Some issuers' WAFs reject non-browser clients (403/405), so these files
    # cannot be fetched by the ingest script and must be dropped into the
    # corpus dir by hand before the build runs.
    manual: bool = False


# Public PDFs, prioritising stable government / NGO endpoints. Auto-download
# URLs are best-effort; if any fail the user is told to drop the file manually
# into the corpus dir. `manual` sources are never fetched and must be present
# before ingestion (enforced by a pre-flight check in `build_rag_corpus`).
SOURCES: list[CorpusSource] = [
    CorpusSource(
        doc_id="EFSA_DRV_SUMMARY",
        title="EFSA Dietary Reference Values - summary report",
        url="https://www.efsa.europa.eu/sites/default/files/2017_09_DRVs_summary_report.pdf",
    ),
    CorpusSource(
        doc_id="NICE_NG28",
        title="NICE NG28: Type 2 diabetes in adults: management",
        url="https://www.nice.org.uk/guidance/ng28/resources/type-2-diabetes-in-adults-management-pdf-1837338615493",
    ),
    CorpusSource(
        doc_id="DGA_2025_2030",
        title="Dietary Guidelines for Americans, 2025-2030",
        url="https://cdn.realfood.gov/DGA.pdf",
    ),
    CorpusSource(
        doc_id="NHLBI_DASH",
        title="NHLBI - Your Guide to Lowering Your Blood Pressure with DASH",
        url="https://www.nhlbi.nih.gov/sites/default/files/publications/WhyDASHWorks_UpdateNov20.pdf",
    ),
    # CorpusSource(
    #     doc_id="ACI_DIET_SPECS",
    #     title="ACI - Therapeutic Diet Specifications for Adult Inpatients",
    #     url="https://aci.health.nsw.gov.au/__data/assets/pdf_file/0006/160557/ACI_AdultDietSpecs-march2017.pdf",
    #     manual=True,
    # ),
    CorpusSource(
        doc_id="NIAID_FOOD_ALLERGY",
        title="NIAID - Guidelines for the Diagnosis and Management of Food Allergy (patient summary)",
        url="https://www.niaid.nih.gov/sites/default/files/faguidelinespatient.pdf",
        manual=True,
    ),
    # CorpusSource(
    #     doc_id="KY_DIABETES_MEAL_PLANNING",
    #     title="KY DPH - Diabetes Meal Planning (Nutrition Basics)",
    #     url="https://www.chfs.ky.gov/agencies/dph/dpqi/cdpb/dpcp/nutrition%20basics.pdf",
    #     manual=True,
    # ),
]
