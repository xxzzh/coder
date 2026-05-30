#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a multi-format sample knowledge corpus to knowledge_base/raw."""

from __future__ import annotations

import textwrap
import zipfile
import zlib
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "knowledge_base" / "raw"

TOPICS = [
    {
        "slug": "water_cycle_hydrology",
        "title": "Water Cycle and Hydrology",
        "summary": "The water cycle describes how water moves among oceans, atmosphere, land, rivers, groundwater, snow, and ice.",
        "facts": [
            "Evaporation changes liquid water into water vapor, mainly from oceans, lakes, rivers, and wet soils.",
            "Transpiration is water vapor released by plants through stomata; together with evaporation it is called evapotranspiration.",
            "Condensation forms cloud droplets when moist air cools to its dew point.",
            "Precipitation includes rain, snow, sleet, and hail that fall from clouds to Earth's surface.",
            "Infiltration moves water from the surface into soil, while percolation moves it deeper toward groundwater.",
            "Runoff flows over land into streams and rivers when rainfall exceeds infiltration or soil storage.",
            "Groundwater is stored in aquifers and can discharge naturally into springs, rivers, lakes, and oceans.",
        ],
        "qa": [
            ("What drives the water cycle?", "Solar energy and gravity drive evaporation, atmospheric movement, precipitation, runoff, and groundwater flow."),
            ("Why is infiltration important?", "It recharges soil moisture and groundwater and reduces the amount of immediate surface runoff."),
        ],
    },
    {
        "slug": "plate_tectonics_earthquakes",
        "title": "Plate Tectonics and Earthquakes",
        "summary": "Plate tectonics explains how Earth's lithosphere is divided into moving plates that interact at boundaries.",
        "facts": [
            "Divergent boundaries form where plates move apart, often creating new oceanic crust at mid-ocean ridges.",
            "Convergent boundaries form where plates move toward each other; subduction can produce trenches, volcanoes, and strong earthquakes.",
            "Transform boundaries form where plates slide past each other, commonly generating shallow earthquakes.",
            "Most earthquakes occur when stress on a fault exceeds friction and stored elastic energy is released suddenly.",
            "The focus is the point inside Earth where rupture begins, and the epicenter is the surface point above the focus.",
            "P waves travel fastest and pass through solids, liquids, and gases; S waves travel only through solids.",
            "Seismic waves and shadow zones help scientists infer the structure of Earth's interior.",
        ],
        "qa": [
            ("What causes many earthquakes?", "Many earthquakes are caused by sudden slip on faults at or near tectonic plate boundaries."),
            ("Why are S waves useful?", "Because S waves do not travel through liquids, their paths provide evidence about liquid layers inside Earth."),
        ],
    },
    {
        "slug": "electric_circuits_basics",
        "title": "Electric Circuits Basics",
        "summary": "An electric circuit is a closed path that allows electric charge to move through components.",
        "facts": [
            "Electric current is the rate of flow of electric charge and is measured in amperes.",
            "Voltage is electric potential difference and is measured in volts.",
            "Resistance opposes current and is measured in ohms.",
            "Ohm's law states that voltage equals current multiplied by resistance: V = I x R.",
            "In a series circuit, the same current flows through each component and resistances add.",
            "In a parallel circuit, components share the same voltage and total resistance decreases as branches are added.",
            "Electrical power is the rate of energy transfer and can be calculated as P = V x I.",
        ],
        "qa": [
            ("What is Ohm's law?", "Ohm's law is V = I x R, relating voltage, current, and resistance."),
            ("What changes in a parallel circuit?", "Each branch has the same voltage, and adding branches usually lowers total resistance."),
        ],
    },
    {
        "slug": "newton_laws_motion",
        "title": "Newton's Laws of Motion",
        "summary": "Newton's laws describe how forces affect the motion of objects in classical mechanics.",
        "facts": [
            "The first law says an object remains at rest or in uniform straight-line motion unless acted on by a net external force.",
            "The second law states that net force equals mass times acceleration: F = m x a.",
            "The third law states that for every action force there is an equal and opposite reaction force.",
            "Mass measures inertia, while weight is the gravitational force acting on mass.",
            "Friction is a contact force that opposes relative motion or the tendency of motion between surfaces.",
            "Momentum equals mass times velocity and is conserved in isolated systems.",
            "Newton's laws work very well for everyday speeds and sizes but are extended by relativity and quantum mechanics in extreme cases.",
        ],
        "qa": [
            ("What does inertia mean?", "Inertia is the tendency of an object to resist changes in its state of motion."),
            ("What is Newton's second law?", "It states that net force equals mass times acceleration, written as F = m x a."),
        ],
    },
    {
        "slug": "dna_genetics_basics",
        "title": "DNA and Genetics Basics",
        "summary": "DNA stores hereditary information used by cells to build and regulate living systems.",
        "facts": [
            "DNA is a polymer made of nucleotides containing a sugar, phosphate group, and nitrogenous base.",
            "The four DNA bases are adenine, thymine, cytosine, and guanine.",
            "Base-pairing rules are adenine with thymine and cytosine with guanine.",
            "A gene is a DNA sequence that contributes to a functional product, often a protein or functional RNA.",
            "DNA replication copies genetic information before cell division.",
            "Transcription copies information from DNA into RNA, and translation uses messenger RNA to build proteins.",
            "Mutations are changes in DNA sequence; their effects can be harmful, neutral, or beneficial depending on context.",
        ],
        "qa": [
            ("What does DNA store?", "DNA stores genetic information in the sequence of its nucleotide bases."),
            ("What are the base-pairing rules?", "A pairs with T, and C pairs with G in double-stranded DNA."),
        ],
    },
    {
        "slug": "ecosystems_energy_flow",
        "title": "Ecosystems and Energy Flow",
        "summary": "An ecosystem includes living organisms and the physical environment with which they interact.",
        "facts": [
            "Producers such as plants and algae convert light energy or chemical energy into organic matter.",
            "Consumers obtain energy by eating producers or other consumers.",
            "Decomposers break down dead organisms and waste, recycling nutrients into the environment.",
            "Energy generally flows one way through food webs, while matter cycles through ecosystems.",
            "Only a fraction of energy is transferred from one trophic level to the next, with much lost as heat.",
            "Biodiversity can improve ecosystem resilience by providing functional redundancy and varied responses to disturbance.",
            "Limiting factors such as nutrients, light, water, temperature, and space influence population growth.",
        ],
        "qa": [
            ("How does energy move in ecosystems?", "Energy flows from producers to consumers and decomposers, with losses as heat at each transfer."),
            ("What do decomposers do?", "They break down dead matter and waste, returning nutrients to the environment."),
        ],
    },
    {
        "slug": "weather_climate_basics",
        "title": "Weather and Climate Basics",
        "summary": "Weather is short-term atmospheric condition, while climate describes long-term patterns and averages.",
        "facts": [
            "Weather includes temperature, humidity, wind, precipitation, cloud cover, and atmospheric pressure.",
            "Climate is commonly described using statistics over decades, often a 30-year reference period.",
            "Air pressure differences help drive winds from higher pressure toward lower pressure areas.",
            "Warm air can hold more water vapor than cold air, which affects cloud formation and precipitation.",
            "Ocean currents transport heat and influence regional climates.",
            "Greenhouse gases absorb and emit infrared radiation, helping regulate Earth's surface temperature.",
            "Climate variability includes natural patterns such as seasonal cycles and ocean-atmosphere oscillations.",
        ],
        "qa": [
            ("What is the difference between weather and climate?", "Weather is short-term atmospheric condition; climate is the long-term pattern of weather statistics."),
            ("Why do winds form?", "Winds form largely because pressure differences and Earth's rotation move air through the atmosphere."),
        ],
    },
    {
        "slug": "periodic_table_chemistry",
        "title": "Periodic Table and Chemistry",
        "summary": "The periodic table organizes elements by atomic number and recurring chemical properties.",
        "facts": [
            "Atomic number equals the number of protons in an atom's nucleus.",
            "Elements in the same group often have similar valence electron patterns and chemical behavior.",
            "Periods are rows in the periodic table and generally correspond to increasing electron shells.",
            "Metals tend to conduct electricity and heat, while nonmetals vary widely in physical properties.",
            "Ionic bonds often form through electron transfer between atoms, producing charged ions.",
            "Covalent bonds involve atoms sharing pairs of electrons.",
            "Chemical reactions rearrange atoms and bonds but do not create or destroy atoms in ordinary chemistry.",
        ],
        "qa": [
            ("What does atomic number mean?", "It is the number of protons in the nucleus of an atom."),
            ("Why are elements grouped?", "Elements in the same group often share similar valence electron structures and chemical properties."),
        ],
    },
    {
        "slug": "algorithms_data_structures",
        "title": "Algorithms and Data Structures",
        "summary": "Algorithms are step-by-step procedures for solving problems, and data structures organize information for efficient use.",
        "facts": [
            "A data structure stores and organizes data to support operations such as search, insertion, deletion, and traversal.",
            "Arrays provide indexed access, while linked lists make some insertions and deletions easier when node references are known.",
            "Stacks follow last-in, first-out order; queues follow first-in, first-out order.",
            "Hash tables use a hash function to map keys to storage locations and often support average-case constant-time lookup.",
            "Trees represent hierarchical relationships, and binary search trees keep keys ordered to support efficient search.",
            "Graphs model relationships among nodes and edges, such as networks, routes, dependencies, and social connections.",
            "Big O notation describes how resource use grows with input size, abstracting away machine-specific constants.",
        ],
        "qa": [
            ("What is Big O notation?", "Big O describes how an algorithm's time or space use grows as input size increases."),
            ("What is a hash table used for?", "It maps keys to values and is commonly used for fast lookup, insertion, and deletion."),
        ],
    },
    {
        "slug": "database_acid_transactions",
        "title": "Database ACID Transactions",
        "summary": "ACID describes properties that help database transactions remain reliable despite errors and concurrency.",
        "facts": [
            "Atomicity means a transaction's operations are treated as an all-or-nothing unit.",
            "Consistency means a committed transaction moves the database from one valid state to another according to constraints.",
            "Isolation controls how concurrent transactions observe each other's intermediate or final effects.",
            "Durability means committed data should survive crashes or power loss, usually through logs, checkpoints, or replication.",
            "A transaction groups one or more operations into a logical unit of work.",
            "Indexes can speed up reads but add storage cost and can slow writes because index entries must be maintained.",
            "Normalization reduces avoidable duplication by organizing data into related tables with well-defined keys.",
        ],
        "qa": [
            ("What does ACID stand for?", "ACID stands for Atomicity, Consistency, Isolation, and Durability."),
            ("Why use indexes?", "Indexes speed up many queries, but they require storage and maintenance during writes."),
        ],
    },
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_docx(path: Path, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraph_xml = "".join(f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>""",
        )
        zf.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{paragraph_xml}<w:sectPr/></w:body></w:document>""",
        )


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def create_xlsx(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strings: list[str] = []
    indexes: dict[str, int] = {}
    for row in rows:
        for value in row:
            if value not in indexes:
                indexes[value] = len(strings)
                strings.append(value)

    rows_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cells.append(f'<c r="{col_name(col_index)}{row_index}" t="s"><v>{indexes[value]}</v></c>')
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    shared_xml = "".join(f"<si><t>{escape(value)}</t></si>" for value in strings)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>""",
        )
        zf.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Knowledge" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>""",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{''.join(rows_xml)}</sheetData></worksheet>""",
        )
        zf.writestr(
            "xl/sharedStrings.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(strings)}" uniqueCount="{len(strings)}">{shared_xml}</sst>""",
        )


def create_pdf(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream_lines = ["BT", "/F1 10 Tf", "54 760 Td"]
    for line in lines:
        for wrapped in textwrap.wrap(line, width=92) or [""]:
            safe = wrapped.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.extend([f"({safe}) Tj", "0 -14 Td"])
    stream_lines.append("ET")
    compressed = zlib.compress("\n".join(stream_lines).encode("latin-1", errors="ignore"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(compressed) + compressed + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    path.write_bytes(b"".join(chunks + xref + [trailer]))


def markdown_for(topic: dict[str, object]) -> str:
    facts = "\n".join(f"- {fact}" for fact in topic["facts"])
    qa = "\n".join(f"- Q: {question}\n  A: {answer}" for question, answer in topic["qa"])
    return f"# {topic['title']}\n\n## Overview\n{topic['summary']}\n\n## Key Facts\n{facts}\n\n## Answerable Questions\n{qa}\n"


def rows_for(topic: dict[str, object]) -> list[list[str]]:
    rows = [["Topic", "Category", "Fact or answer"]]
    rows.append([topic["title"], "Overview", topic["summary"]])
    for index, fact in enumerate(topic["facts"], start=1):
        rows.append([topic["title"], f"Fact {index}", fact])
    for question, answer in topic["qa"]:
        rows.append([topic["title"], f"Question: {question}", answer])
    return rows


def main() -> None:
    for topic in TOPICS:
        paragraphs = [topic["title"], topic["summary"], "Key facts", *topic["facts"], "Answerable questions"]
        paragraphs.extend(f"{question} {answer}" for question, answer in topic["qa"])
        pdf_lines = [topic["title"], topic["summary"], "Key facts:", *topic["facts"], "Answerable questions:"]
        pdf_lines.extend(f"Q: {question} A: {answer}" for question, answer in topic["qa"])

        write_text(RAW_DIR / "markdown" / f"{topic['slug']}.md", markdown_for(topic))
        create_docx(RAW_DIR / "word" / f"{topic['slug']}.docx", paragraphs)
        create_pdf(RAW_DIR / "pdf" / f"{topic['slug']}.pdf", pdf_lines)
        create_xlsx(RAW_DIR / "excel" / f"{topic['slug']}.xlsx", rows_for(topic))

    print(f"Added {len(TOPICS)} documents per format under {RAW_DIR}")


if __name__ == "__main__":
    main()
