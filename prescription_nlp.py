import os
import cv2
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import re
import json
from fpdf import FPDF

# Knowledge base for drug metadata
MEDICINE_KB = {
    "pantoprazole": {
        "side_effects": ["headache", "nausea", "abdominal pain"],
        "age_group": "12 years and above",
        "indications": ["acid reflux", "gastritis", "stomach ulcers"]
    },
    "amoxicillin": {
        "side_effects": ["nausea", "rash", "diarrhea"],
        "age_group": "children and adults",
        "indications": ["bacterial infections", "tonsillitis", "UTI"]
    },
    "vomilast": {
        "side_effects": ["drowsiness", "dry mouth"],
        "age_group": "Children and adults",
        "indications": ["vomiting", "motion sickness"]
    },
    "zoclar": {
        "side_effects": ["nausea", "diarrhea"],
        "age_group": "Adults",
        "indications": ["bacterial infection"]
    },
    "gestakind": {
        "side_effects": ["constipation", "nausea"],
        "age_group": "Adults",
        "indications": ["pregnancy support", "nausea"]
    },
    "abciximab": {
        "side_effects": ["bleeding", "nausea"],
        "age_group": "Adults",
        "indications": ["heart attack", "angioplasty"]
    }
}

FREQUENCY_MAP = {
    "od": "once a day", "bid": "twice a day",
    "tid": "three times a day", "qid": "four times a day",
    "hs": "at bedtime", "sos": "only if needed"
}

ALLERGY_DB = {
    "penicillin": ["amoxicillin", "augmentin", "amoxyclav"],
    "sulfa": ["cotrimoxazole"],
    "aspirin": ["aspirin", "ecosprin", "clopidogrel"]
}

PATIENT_ALLERGIES = ["penicillin"]

def doctr_extract(image_path):
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images([image_path])
    res = model(doc)
    lines = []
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                text_line = "".join(w.value for w in line.words)
                if text_line.strip():
                    lines.append(text_line)
    return lines

def group_prescription_blocks(lines):
    blocks = []
    current = []
    for line in lines:
        if re.match(r"^\d+\)", line):
            if current:
                blocks.append(" ".join(current))
                current = []
        current.append(line.strip())
    if current:
        blocks.append(" ".join(current))
    return blocks

def clean_text(text):
    text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)
    text = text.title()
    text = text.replace("Afterfood", "After Food")
    text = re.sub(r"\b(\d+)\s*Day\b", lambda m: "1 Day" if m.group(1) == "1" else f"{m.group(1)} Days", text, flags=re.IGNORECASE)
    return text

def export_to_pdf(entries, filename="static/prescription_summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="MediGuide AI – Prescription Summary", ln=True, align="C")
    for entry in entries:
        pdf.ln()
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt=entry["drug"], ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, txt=entry["instructions"])
    pdf.output(filename)

def parse_line(line):
    print(f"\n🔹 Parsing Block: {line}")
    t = line.strip().lower()
    t_no_space = t.replace(" ", "")

    drug_match = re.search(r"\d+\)\s*(?:tab|cap|syp|inj)\.?\s*([a-z0-9]+)", t)
    drug = drug_match.group(1) if drug_match else None
    print("  - Drug:", drug if drug else "❌ Not found")
    if not drug:
        return None

    strength_match = re.search(r"(\d+\s*mg)", t)
    strength = strength_match.group(1) if strength_match else None
    print("  - Strength:", strength if strength else "❌ Not found")

    freq_match = re.findall(r"(\d+\s*(?:morning|night|noon|evening)|after\s*food)", t)
    if not freq_match:
        freq_match = re.findall(r"(\d*(morning|night|noon|evening))", t_no_space)
    freq = ", ".join([f if isinstance(f, str) else f[0] for f in freq_match]) if freq_match else None
    print("  - Frequency:", freq if freq else "❌ Not found")

    duration_match = re.search(r"(\d+\s*(?:day|days)|\d+(?:day|days))", t_no_space, re.IGNORECASE)
    duration = duration_match.group(1).capitalize() if duration_match else None
    print("  - Duration:", duration if duration else "❌ Not found")

    kb = MEDICINE_KB.get(drug.lower(), {
        "side_effects": ["Not available"],
        "age_group": "General",
        "indications": ["Not available"]
    })

    side_effects = kb["side_effects"]
    age_group = kb["age_group"]
    indications = kb["indications"]

    aw = None
    for a in PATIENT_ALLERGIES:
        if drug.lower() in ALLERGY_DB.get(a, []):
            aw = f"⚠️ Avoid if allergic to {a.title()}"

    instr = f"{drug.title()} is used to treat {', '.join(indications)}. "
    instr += "Usually taken"
    if strength: instr += f" as {strength}"
    if freq: instr += f", {clean_text(freq)}"
    if duration: instr += f" for {clean_text(duration)}"
    if not (strength or freq or duration): instr += " as directed"
    instr += f". Suitable for age group: {age_group}. "
    instr += f"Possible side effects: {', '.join(side_effects)}."
    if aw: instr += f" {aw}"

    return {
        "drug": drug.title(),
        "strength": strength or "Not mentioned",
        "frequency": clean_text(freq) if freq else "Not mentioned",
        "duration": clean_text(duration) if duration else "Not mentioned",
        "instructions": instr,
        "age_group": age_group,
        "indications": indications,
        "side_effects": side_effects,
        "allergy_warning": aw
    }
