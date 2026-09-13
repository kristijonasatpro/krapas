"""python -m pytest kirciuoklis/test_normalize_lt.py -q   (or run directly)"""
from normalize_lt import normalize as N

CASES = [
    ("Kaina 2334 Eur.", "Kaina du tūkstančiai trys šimtai trisdešimt keturi eurai."),
    ("21 euras, 22 eurai, 30 eurų.", "Dvidešimt vienas euras, dvidešimt du eurai, trisdešimt eurų."),
    ("Turiu 2 dukras ir 3 sūnus.", "Turiu dvi dukras ir tris sūnus."),
    ("iš 5 vaikų", "iš penkių vaikų"),
    ("Apie 1 mln. žmonių.", "Apie vieną milijoną žmonių."),
    ("5,5 % daugiau", "penki su puse procento daugiau"),
    ("3,75 km", "trys kablelis septyniasdešimt penki kilometro"),
    ("2024 m.", "Du tūkstančiai dvidešimt ketvirtais metais."),
    ("2025 m. rugsėjo 7 d. buvo šilta.", "Du tūkstančiai dvidešimt penktų metų rugsėjo septintą dieną buvo šilta."),
    ("Susitinkam 14:30.", "Susitinkam keturiolika trisdešimt."),
    ("9:05", "devynios nulis penkios"),
    ("3-ią vietą", "trečią vietą"),
    ("21-ąją premiją", "dvidešimt pirmąją premiją"),
    ("GPT-5", "GPT penki"),
    ("120 km/h", "šimtas dvidešimt kilometrų per valandą"),
    ("t. y. 12 proc.", "tai yra dvylika procentų."),
    ("Tel. +370 612 34567.", "Telefonas plius trys septyni nulis, šeši vienas du, trys keturi penki šeši septyni."),
    ("Be skaičių.", "Be skaičių."),
]


def test_cases():
    for src, want in CASES:
        got = N(src); assert got == want, f"{src!r}: {got!r} != {want!r}"


if __name__ == "__main__":
    bad = 0
    for src, want in CASES:
        got = N(src); ok = got == want; bad += not ok
        print(("ok  " if ok else "FAIL"), src, "->", got, ("" if ok else f"   (want: {want})"))
    print("failures:", bad)
