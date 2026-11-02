from esasr.normalize import detected_lang_tag, expand_numbers, normalize, strip_lang_tags


def test_case_punct_and_spanish_marks():
    assert normalize("¿Cómo estás? ¡Muy bien!") == "como estas muy bien"


def test_keep_accents_option():
    assert normalize("¿Cómo estás?", fold_accents=False) == "cómo estás"


def test_enye_is_kept_when_folding():
    assert normalize("El niño y la Ñandú; pingüino") == "el niño y la ñandu pinguino"


def test_numbers_to_spanish_words():
    assert normalize("Tengo 25 años") == "tengo veinticinco años"
    assert normalize("1300 dólares") == normalize("mil trescientos dólares")
    assert normalize("1.300 dólares") == "mil trescientos dolares"
    assert normalize("el 15%") == "el quince por ciento"
    assert normalize("3,5 kilos") == "tres coma cinco kilos"


def test_ordinals():
    assert "primero" in expand_numbers("el 1º lugar")
    assert "primera" in expand_numbers("la 1ª vez")


def test_nemotron_lang_tag_removed():
    assert normalize("Buenas tardes. <es-US>") == "buenas tardes"
    assert strip_lang_tags("Hola. <es-ES>") == "Hola."
    assert detected_lang_tag("Hola. <es-ES>") == "es-ES"


def test_annotations_removed():
    assert normalize("bueno [ruido] pues <sil> sí") == "bueno pues si"


def test_english_contractions_survive():
    assert normalize("I don't know, ¿verdad?") == "i don't know verdad"
    assert normalize("‘quote’ and 'single'") == "quote and single"


def test_hyphen_and_symbols():
    assert normalize("e-mail & más") == "e mail y mas"


def test_idempotent():
    s = "¡Hola! Son las 3:45, ¿vale? <es-US>"
    assert normalize(normalize(s)) == normalize(s)


def test_empty():
    assert normalize("") == ""
    assert normalize("¿?¡!") == ""
