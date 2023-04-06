patches = [
    ("Pt_le2i", "Pt_se2i"),
    ("Pt_re2i", "Pt_oe2i"),
    ("Pt_lPe", "Pt_sPe"),
    ("Pt_rPe", "Pt_oPe"),
]


def patch_data(qa):
    for a, b in patches:
        if b in qa:
            continue
        qa[b] = qa[a]
        qa.pop(a)
    return qa


qa = {
    "Pt_le2i": {"name": "Pt_le2i"},
    "Pt_re2i": {"name": "Pt_re2i"},
    "Pt_lPe": {"name": "Pt_lPe"},
    "Pt_rPe": {"name": "Pt_rPe"},
}

print(qa)
print(patch_data(qa))
