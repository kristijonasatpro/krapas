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
    ("Pirmasis vyks antradienį, 14 valandą, Kongresų rūmuose.", "Pirmasis vyks antradienį, keturioliktą valandą, Kongresų rūmuose."),
    ("Dirbame nuo 9 iki 17 valandos.", "Dirbame nuo devintos iki septynioliktos valandos."),
    ("Iki 9 valandos.", "Iki devintos valandos."),
    ("Liko 2 valandos.", "Liko dvi valandos."),
    ("Laukėme 14 valandų.", "Laukėme keturiolika valandų."),
    ("Truko 24 valandas.", "Truko dvidešimt keturias valandas."),
    ("Truko 21 valandą.", "Truko dvidešimt vieną valandą."),
    ("21 valandą bus koncertas.", "Dvidešimt pirmą valandą bus koncertas."),
    ("Trukmė gali būti nuo 1 iki 21 valandos.", "Trukmė gali būti nuo vienos iki dvidešimt vienos valandos."),
    ("Susitinkame 9–10 valandą.", "Susitinkame devintą dešimtą valandą."),
    ("Susitinkame 14\u00a0valandą.", "Susitinkame keturioliktą valandą."),
    ("Susitiksime 1 valandą.", "Susitiksime vieną valandą."),
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
