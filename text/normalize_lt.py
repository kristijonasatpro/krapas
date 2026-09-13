"""Lithuanian text normalisation for TTS: numbers, units, currency, percent, decimals, years, dates, times,
phone numbers and ordinal suffixes become words, so the stresser and the voice never see a digit.

    normalize("Kaina 2334 Eur, t. y. 5,5 % daugiau nei 2024 m.")
    -> "Kaina du tūkstančiai trys šimtai trisdešimt keturi eurai, tai yra penki su puse procento
        daugiau nei du tūkstančiai dvidešimt ketvirtais metais."

Scope (v1): cardinals to 10^12; the unit word 1–9 declines by gender and case guessed from the following
noun (-as/-es/-is/-us -> accusative, -ų -> genitive, feminine endings -> feminine); unit abbreviations
(Eur, €, %, proc., kg, km, m, cm, mm, l, val., min., sek., mln., mlrd., tūkst., d., km/h, GB, USD) get the
right noun form after the number; decimals "3,5" -> "trys su puse", "3,75" -> "trys kablelis septyniasdešimt
penki"; years "2024 m." / "2024 metais" / "2024-aisiais"; dates "2024 m. rugsėjo 7 d."; times "14:30";
ordinals with suffix "3-ias", "21-ąją"; letters-hyphen-number "GPT-5" -> "GPT penki"; phone numbers digit by
digit; common abbreviations (t. y., pvz., ir kt., tel.); a preposition right before the number sets the case
(apie 1 mln. -> apie vieną milijoną, iš 5 vaikų -> iš penkių vaikų). Everything else is left as is; stress
marks are added later by kirtis_v2 (stress_text(..., normalize=True) calls this first).
"""
import re
from num2words import num2words

# unit words 1–9: [gender][case] ; teens/tens/hundreds are left as num2words gives them (nominative)
UNIT = {
    ("m", "nom"): ["", "vienas", "du", "trys", "keturi", "penki", "šeši", "septyni", "aštuoni", "devyni"],
    ("m", "acc"): ["", "vieną", "du", "tris", "keturis", "penkis", "šešis", "septynis", "aštuonis", "devynis"],
    ("m", "gen"): ["", "vieno", "dviejų", "trijų", "keturių", "penkių", "šešių", "septynių", "aštuonių", "devynių"],
    ("f", "nom"): ["", "viena", "dvi", "trys", "keturios", "penkios", "šešios", "septynios", "aštuonios", "devynios"],
    ("f", "acc"): ["", "vieną", "dvi", "tris", "keturias", "penkias", "šešias", "septynias", "aštuonias", "devynias"],
    ("f", "gen"): ["", "vienos", "dviejų", "trijų", "keturių", "penkių", "šešių", "septynių", "aštuonių", "devynių"],
}
DIGIT = ["nulis", "vienas", "du", "trys", "keturi", "penki", "šeši", "septyni", "aštuoni", "devyni"]
ORD_STEM = {1: "pirm", 2: "antr", 3: "treči", 4: "ketvirt", 5: "penkt", 6: "šešt", 7: "septint", 8: "aštunt", 9: "devint", 10: "dešimt",
            11: "vienuolikt", 12: "dvylikt", 13: "trylikt", 14: "keturiolikt", 15: "penkiolikt", 16: "šešiolikt", 17: "septyniolikt",
            18: "aštuoniolikt", 19: "devyniolikt", 20: "dvidešimt", 30: "trisdešimt", 40: "keturiasdešimt", 50: "penkiasdešimt",
            60: "šešiasdešimt", 70: "septyniasdešimt", 80: "aštuoniasdešimt", 90: "devyniasdešimt", 100: "šimt", 1000: "tūkstant"}
# unit abbreviations: (nom sg, nom pl, gen pl, gen sg, gender)
UNITS = {
    "eur": ("euras", "eurai", "eurų", "euro", "m"), "€": ("euras", "eurai", "eurų", "euro", "m"),
    "usd": ("doleris", "doleriai", "dolerių", "dolerio", "m"), "$": ("doleris", "doleriai", "dolerių", "dolerio", "m"),
    "%": ("procentas", "procentai", "procentų", "procento", "m"), "proc.": ("procentas", "procentai", "procentų", "procento", "m"),
    "kg": ("kilogramas", "kilogramai", "kilogramų", "kilogramo", "m"), "g": ("gramas", "gramai", "gramų", "gramo", "m"),
    "km/h": ("kilometras per valandą", "kilometrai per valandą", "kilometrų per valandą", "kilometro per valandą", "m"),
    "m/s": ("metras per sekundę", "metrai per sekundę", "metrų per sekundę", "metro per sekundę", "m"),
    "km": ("kilometras", "kilometrai", "kilometrų", "kilometro", "m"), "m": ("metras", "metrai", "metrų", "metro", "m"),
    "cm": ("centimetras", "centimetrai", "centimetrų", "centimetro", "m"), "mm": ("milimetras", "milimetrai", "milimetrų", "milimetro", "m"),
    "l": ("litras", "litrai", "litrų", "litro", "m"), "ml": ("mililitras", "mililitrai", "mililitrų", "mililitro", "m"),
    "val.": ("valanda", "valandos", "valandų", "valandos", "f"), "min.": ("minutė", "minutės", "minučių", "minutės", "f"),
    "sek.": ("sekundė", "sekundės", "sekundžių", "sekundės", "f"), "s": ("sekundė", "sekundės", "sekundžių", "sekundės", "f"),
    "mln.": ("milijonas", "milijonai", "milijonų", "milijono", "m"), "mlrd.": ("milijardas", "milijardai", "milijardų", "milijardo", "m"),
    "tūkst.": ("tūkstantis", "tūkstančiai", "tūkstančių", "tūkstančio", "m"), "vnt.": ("vienetas", "vienetai", "vienetų", "vieneto", "m"),
    "d.": ("diena", "dienos", "dienų", "dienos", "f"), "gb": ("gigabaitas", "gigabaitai", "gigabaitų", "gigabaito", "m"),
    "mb": ("megabaitas", "megabaitai", "megabaitų", "megabaito", "m"), "kw": ("kilovatas", "kilovatai", "kilovatų", "kilovato", "m"),
    "kwh": ("kilovatvalandė", "kilovatvalandės", "kilovatvalandžių", "kilovatvalandės", "f"),
}
MONTHS_GEN = ["sausio", "vasario", "kovo", "balandžio", "gegužės", "birželio", "liepos", "rugpjūčio", "rugsėjo", "spalio", "lapkričio", "gruodžio"]
ABBR = {"t. y.": "tai yra", "t.y.": "tai yra", "pvz.": "pavyzdžiui", "kt.": "kita", "ir pan.": "ir panašiai", "pan.": "panašiai",
        "ir t. t.": "ir taip toliau", "ir t.t.": "ir taip toliau", "t. t.": "taip toliau", "t.t.": "taip toliau", "žr.": "žiūrėk", "tel.": "telefonas",
        "nr.": "numeris", "pr. Kr.": "prieš Kristų", "e. paštu": "elektroniniu paštu", "el. paštu": "elektroniniu paštu", "š. m.": "šių metų", "š.m.": "šių metų"}
CAP = "A-ZĄČĘĖĮŠŲŪŽ"
ACC_PREP = {"apie", "į", "per", "už", "pro", "prieš", "paskui"}
GEN_PREP = {"iš", "nuo", "iki", "be", "dėl", "prie", "ant", "tarp", "po", "virš", "šalia", "link", "vietoj", "aplink", "greta"}


def acc_of(noun):
    """accusative from the nominative of a unit noun (multi-word units decline the first word)"""
    first, _, rest = noun.partition(" "); rest = " " + rest if rest else ""
    for nom, acc in (("as", "ą"), ("is", "į"), ("ys", "į"), ("us", "ų"), ("a", "ą"), ("ė", "ę"), ("ai", "us"), ("iai", "ius"), ("os", "as"), ("ės", "es")):
        if first.endswith(nom): return first[: -len(nom)] + acc + rest
    return noun


# verbs whose object is accusative: "pasiekė 2334 Eur" -> keturis eurus (stems; looked up within 3 words before the number)
ACC_VERB = re.compile(r"^(pasiek|siek|sudar|gav|gaun|uždirb|mok|sumok|kainav|kainuo|tur|išleid|skyr|skir|investav|investuo|pardav|parduo|pirk|perk|"
                      r"viršij|viršy|atsiėm|surink|renk|prarad|praran|laimėj|laim|gamin|pagamin|eksportav|eksportuo|importav|importuo|užaugin|augin|"
                      r"pritrauk|surinko|sukaup|kaup|padidin|sumažin|numat|planuo|siūl|pasiūl|pateik|patvirtin|priėm|priim)")


def loc_of(noun):
    first, _, rest = noun.partition(" "); rest = " " + rest if rest else ""
    for nom, loc in (("as", "e"), ("is", "yje"), ("ys", "yje"), ("us", "uje"), ("a", "oje"), ("ė", "ėje"), ("iai", "iuose"), ("ai", "uose"), ("os", "ose"), ("ės", "ėse")):
        if first.endswith(nom): return first[: -len(nom)] + loc + rest
    return noun


LOC_PREP = {"apie"} - {"apie"}   # (no preposition takes the locative; the case comes from the noun ending)


def prep_case(t, start):
    """case demanded by a preposition right before the number, else by an accusative-governing verb within 3 words, else None"""
    before = re.findall(r"\w+", t[:start].lower())[-3:]
    if before and before[-1] in ACC_PREP: return "acc"
    if before and before[-1] in GEN_PREP: return "gen"
    for w in reversed(before):
        if w in ACC_PREP or w in GEN_PREP: return None          # a preposition in between governs, not the verb
        if ACC_VERB.match(w): return "acc"
    return None


UNIT[("m", "loc")] = ["", "viename", "dviejuose", "trijuose", "keturiuose", "penkiuose", "šešiuose", "septyniuose", "aštuoniuose", "devyniuose"]
UNIT[("f", "loc")] = ["", "vienoje", "dviejose", "trijose", "keturiose", "penkiose", "šešiose", "septyniose", "aštuoniose", "devyniose"]
BIG = {  # nominative -> (gen, acc, loc)
    "šimtas": ("šimto", "šimtą", "šimte"), "šimtai": ("šimtų", "šimtus", "šimtuose"),
    "tūkstantis": ("tūkstančio", "tūkstantį", "tūkstantyje"), "tūkstančiai": ("tūkstančių", "tūkstančius", "tūkstančiuose"),
    "milijonas": ("milijono", "milijoną", "milijone"), "milijonai": ("milijonų", "milijonus", "milijonuose"),
    "milijardas": ("milijardo", "milijardą", "milijarde"), "milijardai": ("milijardų", "milijardus", "milijarduose"),
}
TEENS = {num2words(i, lang="lt") for i in range(11, 20)}
TENS = {num2words(i, lang="lt") for i in range(10, 100, 10)}


def _decline(word, gender, case, is_last):
    """one token of a num2words cardinal in the requested case; gender applies to the final unit only"""
    if case == "nom": return word
    if word in BIG: return BIG[word][("gen", "acc", "loc").index(case)]
    if word in TEENS: return word[:-1] + {"gen": "os", "acc": "ą", "loc": "oje"}[case]
    if word in TENS: return word + ({"gen": "ies", "acc": "", "loc": "yje"}[case] if is_last else "")   # dvidešimties, but dvidešimt trijų
    m = UNIT[("m", "nom")]
    if word in m:
        u = m.index(word); return UNIT[(gender if is_last else "m", case)][u]
    return word


def cardinal(n, gender="m", case="nom"):
    """num2words with 'vienas šimtas' -> 'šimtas' etc., every token declined for gen/acc/loc, the final
    unit word (1–9) also by gender: cardinal(2334, 'm', 'acc') -> 'du tūkstančius tris šimtus trisdešimt keturis'"""
    w = num2words(n, lang="lt")
    w = re.sub(r"\bvienas (šimtas|tūkstantis|milijonas|milijardas)\b", r"\1", w)     # tūkstantis šimtas penkiasdešimt trys
    parts = w.split(" ")
    if gender == "f" and case == "nom":
        u = n % 10
        if 1 <= u <= 9 and not 11 <= n % 100 <= 19 and parts[-1] == UNIT[("m", "nom")][u]: parts[-1] = UNIT[("f", "nom")][u]
        return " ".join(parts)
    return " ".join(_decline(x, gender, case, i == len(parts) - 1) for i, x in enumerate(parts))


def noun_form(n, unit, case="nom"):
    sg, pl, gen_pl, gen_sg, g = unit
    if 11 <= n % 100 <= 19 or n % 10 == 0: return gen_pl
    if case == "gen": return gen_sg if n % 10 == 1 else gen_pl
    if case == "acc": return acc_of(sg) if n % 10 == 1 else acc_of(pl)
    if case == "loc": return loc_of(sg) if n % 10 == 1 else loc_of(pl)
    return sg if n % 10 == 1 else pl


def ordinal(n, ending):
    """ordinal of n (< 10^6) with the given adjective ending; only the last component becomes ordinal"""
    if ending.startswith("i") and n % 10 == 3 and not 11 <= n % 100 <= 19: ending = ending[1:]   # treči + ias -> trečias
    if n in ORD_STEM: return ORD_STEM[n] + ending
    if n < 100:
        return f"{cardinal(n - n % 10)} {ORD_STEM[n % 10]}{ending}"
    if n < 1000:
        h, rest = n // 100, n % 100
        if not rest: return f"{ORD_STEM[100]}{ending}" if h == 1 else f"{cardinal(h)} {ORD_STEM[100]}{ending}"
        return f"{'šimtas' if h == 1 else cardinal(h) + ' šimtai'} {ordinal(rest, ending)}"
    th, rest = n // 1000, n % 1000
    if not rest: return f"{ORD_STEM[1000]}{ending}" if th == 1 else f"{cardinal(th)} {ORD_STEM[1000]}{ending}"
    return f"{'tūkstantis' if th == 1 else cardinal(th) + ' tūkstančiai'} {ordinal(rest, ending)}"


def _int(s): return int(re.sub(r"[  ]", "", s))


def _dot(m):
    """an abbreviation dot that also closed the sentence must survive as a period"""
    rest = m.string[m.end():]
    return "." if (rest == "" or re.match(r"\s*$|\s+[" + CAP + "„\"]", rest)) else ""


def noun_case(n, nxt):
    """gender + case of the numeral from the ending of the noun that follows it (heuristic)"""
    w = nxt.lower(); one = n % 10 == 1 and not 11 <= n % 100 <= 19
    if one:                                                            # singular noun: vienas euras, viena diena, vieną dieną, vieno euro
        if w.endswith(("as", "is", "ys", "us")): return "m", "nom"
        if w.endswith(("a", "ė")): return "f", "nom"
        if w.endswith(("ą", "ę", "į")): return ("f" if w.endswith(("ą", "ę")) and not w.endswith("ią") else "m"), "acc"
        if w.endswith(("o", "io")): return "m", "gen"
        if w.endswith(("os", "ės")): return "f", "gen"
        return "m", "nom"
    if w.endswith(("uose", "iuose")): return "m", "loc"                # penkiuose namuose
    if w.endswith(("ose", "ėse", "yse")): return "f", "loc"              # penkiose minutėse
    if w.endswith("ų"): return "m", "nom"                              # "5 vaikų": the noun's -ų is agreement, not the numeral's case (prepositions decide)
    if w.endswith(("us", "ius")): return "m", "acc"                    # tris sūnus
    if w.endswith(("as", "es", "is")): return "f", "acc"               # dvi dukras, tris moteris (after 2+: -as = fem acc pl)
    if re.search(r"(a|ė|ą|ę|os|ės|oms|ėms|omis|ėmis|ose|ėse|ims|imis|yse)$", w): return "f", "nom"
    return "m", "nom"


def normalize(text, initials="letter", range_sep=" "):
    """initials before a surname: 'letter' (default) keeps the letter and drops the dot — "M Sinkevičius", the form
    Regina's training text uses, so the model reads the letter name without ending the sentence; 'drop' removes
    the initial; 'keep' leaves "M." as written (the dot makes the voice pause as at a sentence end)"""
    t = text
    # parentheticals: short ones (acronym expansions, asides) are dropped — read aloud they break the intonation;
    # long ones keep their text without the brackets
    t = re.sub(r"\s*\((?:[^()]{0,40})\)", "", t)
    t = re.sub(r"\(([^()]+)\)", r"\1", t)
    t = re.sub(r"\bir kt\.(?=\s*(?:[,.;:)”“]|$)|\s+[" + CAP + r"„])", lambda m: "ir kita" + _dot(m), t); t = re.sub(r"\bir kt\.", "ir kitas", t)   # end of a list (also sentence end) / before a noun
    for k, v in sorted(ABBR.items(), key=lambda kv: -len(kv[0])):
        # an abbreviation that ends with "." and closes the sentence keeps its period: "ir t. t." -> "ir taip toliau."
        t = re.sub(r"(?<![\w])" + re.escape(k) + r"(?![\w])", lambda m, v=v, k=k: (v[:1].upper() + v[1:] if m.group()[:1].isupper() else v) + (_dot(m) if k.endswith(".") else ""), t, flags=re.I)
    # initials before a surname are dropped: "A. Aleksandravičius" -> "Aleksandravičius"
    if initials == "drop":
        t = re.sub(r"\b[" + CAP + r"]\.\s?(?=[" + CAP + r"]\.\s?[" + CAP + r"][a-ząčęėįšųūž])", "", t)
        t = re.sub(r"\b[" + CAP + r"]\.\s?(?=[" + CAP + r"][a-ząčęėįšųūž])", "", t)
    elif initials == "letter":
        t = re.sub(r"\b([" + CAP + r"])\.\s?(?=[" + CAP + r"]\.\s?[" + CAP + r"][a-ząčęėįšųūž])", r"\1 ", t)
        t = re.sub(r"\b([" + CAP + r"])\.\s?(?=[" + CAP + r"][a-ząčęėįšųūž])", r"\1 ", t)
    NUM = r"\d{1,3}(?:[  ]\d{3})+|\d+"
    MON = "|".join(MONTHS_GEN)
    # dates: 2024 m. rugsėjo 7 d. ; rugsėjo 7 d.
    t = re.sub(r"\b(\d{4}) m\. (" + MON + r") (\d{1,2}) d\.", lambda m: f"{ordinal(_int(m.group(1)), 'ų')} metų {m.group(2)} {ordinal(_int(m.group(3)), 'ą')} dieną{_dot(m)}", t)
    t = re.sub(r"\b(" + MON + r") (\d{1,2}) d\.", lambda m: (f"{m.group(1)} {ordinal(_int(m.group(2)), 'os')} dienos" if prep_case(m.string, m.start()) == "gen"
                                                                 else f"{m.group(1)} {ordinal(_int(m.group(2)), 'ą')} dieną") + _dot(m), t)
    t = re.sub(r"\b(\d{4}) m\.(?=\s+(?:pirm|antr|treči|ketvirt|pabaig|pradži|vidur|sausi|vasari|kov|balandži|geguž|birželi|liep|rugpjū|rugsėj|spali|lapkriči|gruodži))",
               lambda m: f"{ordinal(_int(m.group(1)), 'ų')} metų", t)
    # years
    t = re.sub(r"\b(\d{4}) m\.", lambda m: f"{ordinal(_int(m.group(1)), 'ais')} metais{_dot(m)}", t)
    t = re.sub(r"\b(\d{4}) (metais|metų|metai|metams|metuose)\b", lambda m: f"{ordinal(_int(m.group(1)), {'metais': 'ais', 'metų': 'ų', 'metai': 'i', 'metams': 'iems', 'metuose': 'uose'}[m.group(2)])} {m.group(2)}", t)
    # ordinal suffixes: 3-ias, 5-oji, 21-ąją, 2024-aisiais — the writer's suffix combines with the stem as written
    t = re.sub(r"\b(\d+)-([a-ząčęėįšųūž]{1,8})\b", lambda m: ordinal(_int(m.group(1)), m.group(2)), t)
    # letters-hyphen-number: GPT-5 -> GPT penki
    t = re.sub(r"\b([A-Za-z" + CAP + r"]+)-(\d+)\b", lambda m: f"{m.group(1)} {cardinal(int(m.group(2)))}", t)
    # times 14:30 -> keturiolika trisdešimt ; 9:05 -> devynios nulis penkios ; 10:00 -> dešimt valandų
    def clock(m):
        h, mi, full = int(m.group(1)), int(m.group(2)), bool(m.group(3))
        if mi == 0: return f"{cardinal(h, 'f')} {noun_form(h, UNITS['val.'])}" if h else "nulis valandų"
        if full: return f"{cardinal(h, 'f')} {noun_form(h, UNITS['val.'])} {cardinal(mi, 'f')} {noun_form(mi, UNITS['min.'])}"   # 20:15 val. -> dvidešimt valandų penkiolika minučių
        return f"{cardinal(h, 'f')} {'nulis ' if mi < 10 else ''}{cardinal(mi, 'f')}"
    t = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b([  ]?val\.)?", clock, t)
    t = re.sub(r"\b(\d{1,3}):(\d{1,3})\b", lambda m: f"{cardinal(int(m.group(1)))} – {cardinal(int(m.group(2)))}", t)   # sports score
    # phone numbers: digit by digit
    def phone(m): return ("plius " if m.group().lstrip().startswith("+") else "") + ", ".join(" ".join(DIGIT[int(c)] for c in g) for g in re.findall(r"\d+", m.group()))
    t = re.sub(r"\+\d[\d  ]{6,}\d|(?<=[Tt]elefonas )\d[\d  ]{6,}\d|\b8[  ]\d{3}[  ]\d{5}\b", phone, t)
    # decimals: 3,5 -> trys su puse ; 3,75 -> trys kablelis septyniasdešimt penki ; 0,05 -> nulis kablelis nulis penki ; + unit in genitive singular
    unit_alt = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    def dec(m):
        a, b, u, c = _int(m.group(1)), m.group(2), (m.group(3) or "").lower(), (m.group(4) or "").lower()
        if b == "5" and a == 1: out = "pusantros" if u in UNITS and UNITS[u][4] == "f" else "pusantro"
        elif b == "5": out = f"{cardinal(a)} su puse"
        else:
            frac = " ".join(DIGIT[int(c)] for c in b) if b.startswith("0") else cardinal(int(b))
            out = f"{cardinal(a)} kablelis {frac}"
        if u not in UNITS: return out
        if c in UNITS: return f"{out} {UNITS[u][3]} {UNITS[c][2]}"                      # 1,2 mlrd. Eur -> ... milijardo eurų
        return f"{out} {UNITS[u][3]}{_dot(m) if u.endswith('.') else ''}"
    t = re.sub(r"\b(" + NUM + r")[,.](\d+)(?:[  ]?(" + unit_alt + r")(?:[  ]?(eur|€|usd|\$))?)?(?![\w])", dec, t, flags=re.I)
    # big unit + currency: 3 mln. Eur -> tris milijonus eurų ; 120 tūkst. USD -> šimtas dvidešimt tūkstančių dolerių
    def big_cur(m):
        n, u, c = _int(m.group(1)), m.group(2).lower(), m.group(3).lower(); case = prep_case(m.string, m.start()) or "nom"
        return f"{cardinal(n, 'm', case)} {noun_form(n, UNITS[u], case)} {UNITS[c][2]}"
    t = re.sub(r"\b(" + NUM + r")[  ]?(mln\.|mlrd\.|tūkst\.)[  ]?(eur|€|usd|\$)(?![\w])", big_cur, t, flags=re.I)
    # integer + unit abbreviation: 2334 Eur -> ... keturi eurai ; 12 proc. -> dvylika procentų
    def num_unit(m):
        n, u = _int(m.group(1)), m.group(2).lower(); unit = UNITS[u]; case = prep_case(m.string, m.start()) or "nom"
        return f"{cardinal(n, unit[4], case)} {noun_form(n, unit, case)}{_dot(m) if u.endswith('.') else ''}"
    t = re.sub(r"\b(" + NUM + r")[  ]?(" + unit_alt + r")(?![\w])", num_unit, t, flags=re.I)
    # ranges "8–13 laipsnių": both numbers take the case of the phrase, dash kept (read as a pause)
    def rng(m):
        a, b, nxt = _int(m.group(1)), _int(m.group(2)), m.group(3)
        g, c = noun_case(b, nxt) if nxt else ("m", "nom")
        pc = prep_case(m.string, m.start())
        if pc and c in ("nom", "gen"): c = pc
        return f"{cardinal(a, g, c)}{range_sep}{cardinal(b, g, c)}" + (" " + nxt if nxt else "")     # default no dash: the voice makes a long pause on it
    t = re.sub(r"(?<![\w-])(" + NUM + r")[  ]?[–-][  ]?(" + NUM + r")(?:[  ]+([^\W\d_]+))?", rng, t)
    # remaining integers: gender/case from the following noun
    def num(m):
        n, nxt = _int(m.group(1)), m.group(2)
        g, c = noun_case(n, nxt) if nxt else ("m", "nom")
        pc = prep_case(m.string, m.start())
        if pc and c in ("nom", "gen"): c = pc                          # iš/apie/pasiekė decide; a noun ending in -as/-us/-ose keeps its own case
        return cardinal(n, g, c) + (" " + nxt if nxt else "")
    t = re.sub(r"(?<![\w-])(" + NUM + r")(?:[  ]+([^\W\d_]+))?", num, t)
    t = re.sub(r"[  ]{2,}", " ", t)
    t = re.sub(r"([.!?]\s+)([a-ząčęėįšųūž])", lambda m: m.group(1) + m.group(2).upper(), t)      # a number that opened a sentence
    if text[:1].isdigit() and re.search(r"[.!?]$", text.strip()): t = t[:1].upper() + t[1:]      # ... including the first one (fragments stay as they are)
    return t


if __name__ == "__main__":
    import sys
    tests = ["Kaina 2334 Eur, t. y. 5,5 % daugiau nei 2024 m.", "Turiu 2 dukras ir 3 sūnus, iš 5 vaikų.", "21 euras, 22 eurai, 25 eurų, 111 eurų, 2334 eurų.",
             "Susitinkam 14:30, o rytoj 9:05 arba 10:00.", "2025 m. rugsėjo 7 d. buvo šilta. 2025 metais taip pat.", "Tel. +370 612 34567.",
             "Gavome 3-ią vietą ir 21-ąją premiją 2024-aisiais.", "GPT-5 ir Gemini 2,5 Pro; 0,05 proc.", "Apie 1 mln. žmonių, 12 proc. daugiau.",
             "Nuotolis 3,75 km, greitis 120 km/h.", "Pirko 4 knygas ir 2 lentynas už 1 200 eurų.", "Video truko 45 min., o 2 val. seminaras."]
    for s in sys.argv[1:] or tests:
        print(s, "\n  ->", normalize(s))
