"""Source-agnostic relevance filter for tourism sentiment.

Marks each record with `relevance_kept` (True/False) based on whether the text
looks like it's about travel/tourism/the destination — rather than chit-chat
addressed to the video's creator, one-word reactions, or pure emoji.

We MARK, never drop: filtered-out rows stay in the store/CSV so the filter can
be reviewed. `kept(records)` is a convenience for the ones that passed.
"""

import re

# Travel/tourism intent words and phrases (substring match on lowercased text).
# English "visit"/"tour" already cover French "visiter"/"tourisme" by substring,
# but accented and Arabic forms share no stem with them and need listing.
KEYWORDS = (
    # English
    "visit", "travel", "trip", "tour", "tourist", "tourism", "safari",
    "vacation", "holiday", "itinerary", "flight", "fly to", "hotel",
    "accommodation", "hostel", "backpack", "road trip", "beach", "wildlife",
    "scenery", "landscape", "destination", "bucket list", "sightseeing",
    "going to", "want to go", "wanna go", "been there", "been to", "planning",
    "plan to", "i'm going", "im going", "we went", "i went", "stayed in",
    "safe to", "is it safe", "how safe", "beautiful", "stunning", "gorgeous",
    "must visit", "worth visiting", "can't wait to", "cant wait to",
    # French
    "voyage", "vacances", "séjour", "sejour", "plage", "hôtel", "magnifique",
    "superbe", "découvrir", "decouvrir", "envie d'aller", "je veux aller",
    "j'ai visité", "beau pays", "aller au", "partir au", "c'est sûr",
    "est-ce que c'est sûr",
    # Arabic
    "سياحة", "سافر", "السفر", "زيارة", "أزور", "زرت", "رحلة", "فندق",
    "شاطئ", "جميل", "رائع", "آمنة", "هل هي آمنة",
)

# Destination/place names — a strong signal the comment is about the place.
# Matched as substrings, so entries must not appear inside unrelated words:
# bare "nile" would hit "juvenile", "fes" would hit "manifesto", "limbe" would
# hit "climbers". Where a name is short or ambiguous, qualify it instead.
PLACES = (
    # South Africa
    "south africa", "cape town", "johannesburg", "joburg", "jozi", "durban",
    "kruger", "garden route", "pretoria", "soweto", "table mountain",
    "kzn", "stellenbosch", "drakensberg", "knysna", "port elizabeth",
    "western cape", "sun city", "robben island",
    # Rwanda
    "rwanda", "kigali", "volcanoes national park", "nyungwe", "lake kivu",
    "akagera", "musanze", "ruhengeri", "gisenyi", "rubavu", "virunga",
    "gorilla trekking", "bisoke", "karisimbi", "huye", "land of a thousand hills",
    # Kenya
    "kenya", "nairobi", "mombasa", "maasai mara", "masai mara", "diani",
    "lamu", "amboseli", "nakuru", "lake nakuru", "tsavo", "malindi",
    "rift valley", "mount kenya", "watamu", "samburu",
    # Tanzania
    "tanzania", "zanzibar", "serengeti", "kilimanjaro", "ngorongoro",
    "dar es salaam", "arusha", "stone town", "dodoma", "mafia island",
    "tarangire", "pemba", "moshi", "selous",
    # Egypt
    "egypt", "cairo", "giza", "pyramid", "sphinx", "luxor", "aswan",
    "the nile", "nile river", "nile cruise", "felucca", "sharm el sheikh",
    "sharm", "hurghada", "alexandria", "red sea", "valley of the kings",
    "abu simbel", "dahab", "siwa", "karnak", "egyptian museum",
    "khan el khalili", "coptic cairo", "grand egyptian museum",
    "مصر", "القاهرة", "الأهرامات", "الاهرامات", "الجيزة", "الأقصر", "أسوان",
    "شرم الشيخ", "الغردقة", "الإسكندرية", "البحر الأحمر", "النيل",
    # Senegal
    "senegal", "sénégal", "dakar", "goree", "gorée", "saint-louis",
    "saint louis", "lake retba", "lac rose", "pink lake", "casamance",
    "sine saloum", "thiès", "ziguinchor", "djoudj", "african renaissance",
    # Ghana
    "ghana", "accra", "kumasi", "cape coast", "kakum", "elmina", "takoradi",
    "tamale", "labadi", "mole national park", "lake volta", "volta region",
    "osu castle", "aburi",
    # Nigeria — bare "kano" is unusable (it is inside "volcano"), so the city is
    # listed only in its qualified form.
    "nigeria", "lagos", "abuja", "calabar", "benin city", "ibadan",
    "port harcourt", "kano state", "kano city", "yankari", "obudu",
    "olumo rock", "zuma rock", "idanre", "lekki", "badagry", "niger delta",
    "oshogbo", "osogbo", "eko atlantic", "nike art", "erin ijesha",
    # Zimbabwe
    "zimbabwe", "harare", "victoria falls", "vic falls", "bulawayo", "hwange",
    "matobo", "matopos", "mana pools", "lake kariba", "kariba dam", "nyanga",
    "eastern highlands", "chimanimani", "gonarezhou", "mutare", "masvingo",
    "chinhoyi", "zambezi", "great zimbabwe",
    # Morocco
    "morocco", "maroc", "marrakech", "marrakesh", "casablanca", "fez",
    "chefchaouen", "rabat", "tangier", "essaouira", "merzouga", "sahara",
    "atlas mountains", "agadir", "meknes", "ouarzazate", "jemaa el",
    "hassan ii", "ait ben haddou",
    "المغرب", "مراكش", "الدار البيضاء", "الرباط", "طنجة", "الصحراء",
    # Cameroon
    "cameroon", "cameroun", "yaounde", "yaoundé", "douala", "kribi",
    "mount cameroon", "limbe beach", "bamenda", "buea", "foumban", "waza",
    "korup", "dschang",
)

# Phrases that mark a comment as directed at the creator / low-signal chatter.
CREATOR_PHRASES = (
    "your video", "your videos", "your channel", "your content", "you look",
    "are you single", "great work", "great video", "nice video", "love your",
    "loved your", "subscribed", "subscriber", "first comment", "early gang",
    "who's watching", "whos watching", "notification", "more videos",
    "keep it up", "well done", "thanks for sharing", "thank you for sharing",
    "what camera", "what's your", "whats your", "marry me", "you're beautiful",
    "youre beautiful", "you are beautiful", "i love you",
)


def _words(text: str):
    # [^\W\d_] matches a letter in ANY script, so Arabic and accented French
    # tokenize instead of collapsing to zero words and being dropped by the
    # length guard in is_relevant().
    return re.findall(r"[^\W\d_]+", text.lower())


def is_relevant(text: str) -> bool:
    """Heuristic: is this comment likely about travel/tourism/the destination?"""
    if not text:
        return False
    lowered = text.lower()
    words = _words(lowered)

    # One word or pure emoji/punctuation -> not useful.
    if len(words) < 2:
        return False

    mentions_place = any(place in lowered for place in PLACES)
    has_keyword = any(kw in lowered for kw in KEYWORDS)
    creator_directed = any(phrase in lowered for phrase in CREATOR_PHRASES)

    # Mentioning the destination by name is a strong keep signal, even if the
    # comment also greets the creator.
    if mentions_place:
        return True
    # Otherwise drop creator-directed chatter; keep clear travel-intent text.
    if creator_directed:
        return False
    return has_keyword


def mark(records):
    """Set `relevance_kept` on each record in place. Returns the same list."""
    for rec in records:
        rec["relevance_kept"] = is_relevant(rec.get("text") or "")
    return records


def kept(records):
    """Return only the records that passed the relevance filter."""
    return [r for r in records if r.get("relevance_kept")]
