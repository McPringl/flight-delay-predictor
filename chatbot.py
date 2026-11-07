"""
A rule-based chatbot for flight-delay questions.

It is a conversation, not a form:
  * you can say things in any order ("delta from atlanta to LA tomorrow at 9am"),
  * it remembers the flight from one message to the next,
  * it asks for whatever is still missing,
  * once it knows the flight it gives the answer, and you can follow up:
    "what if it leaves 2 hours later?", "try friday", "why?", "start over".

No external AI service is used: the understanding is done with patterns,
so it handles common phrasing well but not every possible sentence.
"""

import re
from datetime import date, datetime, time, timedelta
from typing import Callable

# ----------------------------------------------------------------------------
# Vocabulary: city names, airline names, months, weekdays
# ----------------------------------------------------------------------------
CITY_TO_AIRPORT = {
    "atlanta": "ATL", "los angeles": "LAX", "new york": "JFK", "newark": "EWR",
    "laguardia": "LGA", "la guardia": "LGA", "chicago": "ORD", "midway": "MDW",
    "dallas": "DFW", "houston": "IAH", "denver": "DEN", "san francisco": "SFO",
    "seattle": "SEA", "miami": "MIA", "boston": "BOS", "las vegas": "LAS",
    "phoenix": "PHX", "orlando": "MCO", "washington": "DCA", "dulles": "IAD",
    "minneapolis": "MSP", "detroit": "DTW", "philadelphia": "PHL", "charlotte": "CLT",
    "san diego": "SAN", "portland": "PDX", "salt lake": "SLC", "nashville": "BNA",
    "austin": "AUS", "tampa": "TPA", "honolulu": "HNL", "san jose": "SJC",
    "new orleans": "MSY", "kansas city": "MCI", "st louis": "STL", "saint louis": "STL",
    "pittsburgh": "PIT", "raleigh": "RDU", "baltimore": "BWI", "fort lauderdale": "FLL",
    "oakland": "OAK", "orange county": "SNA", "indianapolis": "IND", "sacramento": "SMF",
}

# Airline names people actually say -> code.
AIRLINE_WORDS = {
    "american": "AA", "delta": "DL", "united": "UA", "southwest": "WN",
    "alaska": "AS", "jetblue": "B6", "jet blue": "B6", "spirit": "NK",
    "frontier": "F9", "hawaiian": "HA", "allegiant": "G4", "envoy": "MQ",
    "skywest": "OO", "republic": "YX", "mesa": "YV", "psa": "OH",
    "endeavor": "9E", "expressjet": "EV",
}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))

MIN_DATE = date(2023, 1, 1)
MAX_DATE = date(2030, 12, 31)


# ----------------------------------------------------------------------------
# 1. Understanding one message
# ----------------------------------------------------------------------------
def find_airline(text: str, carriers: list[str]) -> str | None:
    lower = text.lower()
    for word, code in AIRLINE_WORDS.items():
        if code in carriers and re.search(r"\b" + re.escape(word) + r"\b", lower):
            return code
    # Codes such as "AA" or "9E": only when written in capitals, so "as" or "mesa" in normal text are ignored.
    for token in re.findall(r"\b[A-Z0-9]{2}\b", text):
        if token in carriers:
            return token
    return None


def find_flight_number(text: str, carriers: list[str]) -> tuple[int | None, str | None]:
    """Return (flight number, carrier code if it was written next to the number)."""
    m = re.search(r"\b([A-Z0-9]{2})\s?(\d{1,4})\b", text)
    if m and m.group(1) in carriers:
        return int(m.group(2)), m.group(1)
    m = re.search(r"\b(?:flight|fl)\s*#?\s*(\d{1,4})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), None
    return None, None


def find_airports(text: str, airports: set[str]) -> list[tuple[int, str]]:
    """Every airport mentioned, as (position in text, code), in order of appearance."""
    lower = text.lower()
    found: list[tuple[int, str]] = []
    for city, code in CITY_TO_AIRPORT.items():
        if code not in airports:
            continue
        for m in re.finditer(r"\b" + re.escape(city) + r"\b", lower):
            found.append((m.start(), code))
    for m in re.finditer(r"\b[A-Za-z]{3}\b", text):
        token = m.group(0)
        if token.upper() not in airports:
            continue
        before = lower[:m.start()]
        after_keyword = re.search(r"\b(from|to)\s*$", before) is not None
        if token.isupper() or after_keyword:      # "ATL" or "to lax", not "the" or "see"
            found.append((m.start(), token.upper()))
    return sorted(found)


def assign_airports(text: str, airports: set[str]) -> dict:
    """Work out which airport is the origin and which is the destination."""
    mentions = find_airports(text, airports)
    if not mentions:
        return {}
    lower = text.lower()
    from_m = re.search(r"\bfrom\b", lower)
    # Use the "to" that comes before a destination airport, so "I want to fly ... to Seattle" works.
    to_positions = [m.start() for m in re.finditer(r"\bto\b", lower)
                    if any(p > m.start() for p, _ in mentions)
                    and (from_m is None or m.start() > from_m.start())]
    to_pos = to_positions[0] if to_positions else None
    result = {}

    if from_m and to_pos is not None:
        origin_candidates = [c for p, c in mentions if from_m.start() < p < to_pos]
        if not origin_candidates:
            origin_candidates = [c for p, c in mentions if p < to_pos]
        dest_candidates = [c for p, c in mentions if p > to_pos]
        if origin_candidates:
            result["origin"] = origin_candidates[0]
        if dest_candidates:
            result["dest"] = next((c for c in dest_candidates if c != result.get("origin")), None)
    elif from_m:
        candidates = [c for p, c in mentions if p > from_m.start()]
        if candidates:
            result["origin"] = candidates[0]
    elif to_pos is not None:
        # "ATL to LAX": airport before "to" is the origin, after it the destination.
        before = [c for p, c in mentions if p < to_pos]
        after = [c for p, c in mentions if p > to_pos]
        if before:
            result["origin"] = before[0]
        if after:
            result["dest"] = next((c for c in after if c != result.get("origin")), None)
    else:
        codes = [c for _, c in mentions]
        result["origin"] = codes[0]
        if len(codes) > 1 and codes[1] != codes[0]:
            result["dest"] = codes[1]
    return {k: v for k, v in result.items() if v}


def find_date(text: str, today: date) -> date | None:
    lower = text.lower()

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lower)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"\b(" + MONTH_PATTERN + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", lower)
    if m:
        return _with_year(MONTHS[m.group(1)], int(m.group(2)), m.group(3), today)

    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + MONTH_PATTERN + r")\.?(?:,?\s+(\d{4}))?\b", lower)
    if m:
        return _with_year(MONTHS[m.group(2)], int(m.group(1)), m.group(3), today)

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", lower)      # US format: month/day
    if m:
        return _with_year(int(m.group(1)), int(m.group(2)), m.group(3), today)

    if "day after tomorrow" in lower:
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", lower):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b", lower):
        return today
    m = re.search(r"\bin (\d{1,3}) days?\b", lower)
    if m:
        return today + timedelta(days=int(m.group(1)))
    for i, name in enumerate(WEEKDAYS):
        if re.search(r"\b" + name + r"\b", lower):
            return today + timedelta(days=(i - today.weekday()) % 7)   # this coming weekday
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _with_year(month: int, day: int, year_text: str | None, today: date) -> date | None:
    """Without a year, use the next time that date comes around."""
    if year_text:
        return _safe_date(int(year_text), month, day)
    candidate = _safe_date(today.year, month, day)
    if candidate and candidate < today:
        candidate = _safe_date(today.year + 1, month, day)
    return candidate


def find_time(text: str) -> time | None:
    lower = text.lower()
    if re.search(r"\bnoon\b", lower):
        return time(12, 0)
    if re.search(r"\bmidnight\b", lower):
        return time(0, 0)
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", lower)
    if m:
        hour = int(m.group(1)) % 12
        if m.group(3).startswith("p"):
            hour += 12
        minute = int(m.group(2) or 0)
        if hour < 24 and minute < 60:
            return time(hour, minute)
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lower)     # 24-hour, e.g. 14:35
    if m:
        return time(int(m.group(1)), int(m.group(2)))
    return None


def find_hour_shift(text: str) -> int | None:
    """'2 hours later' -> +2, '90 minutes earlier' -> -90 (in minutes)."""
    m = re.search(r"\b(\d{1,2})\s*(hours?|hrs?|minutes?|mins?)\s*(later|after|earlier|before)\b", text.lower())
    if not m:
        return None
    amount = int(m.group(1)) * (60 if m.group(2).startswith("h") else 1)
    return amount if m.group(3) in ("later", "after") else -amount


# ----------------------------------------------------------------------------
# 2. The conversation
# ----------------------------------------------------------------------------
REQUIRED = ["carrier", "origin", "dest", "date", "time"]
QUESTIONS = {
    "carrier": "Which airline is the flight with? (for example Delta, United or AA)",
    "origin": "Where are you flying **from**? (city or airport code, e.g. Atlanta or ATL)",
    "dest": "Where are you flying **to**?",
    "date": "What date is the flight? (for example tomorrow, Nov 20, or 2026-11-20)",
    "time": "What time is it scheduled to leave? (for example 9:30 am or 14:35)",
}


class FlightChat:
    """Holds one conversation. Call reply() with each message the user types."""

    def __init__(self, carriers: list[str], airports: list[str],
                 predict: Callable[[dict], dict], airline_names: dict[str, str],
                 today_fn: Callable[[], date] = date.today):
        self.carriers = carriers
        self.airports = set(airports)
        self.predict = predict
        self.airline_names = airline_names
        self.today_fn = today_fn
        self.reset()

    # ---- public --------------------------------------------------------
    def greeting(self) -> str:
        return ("Hi! I can estimate whether a US domestic flight will arrive **15+ minutes late**. "
                "Tell me about your flight in any words, for example:\n\n"
                "> *Delta from Atlanta to LAX tomorrow at 9:30 am*\n\n"
                "I'll ask for anything I'm missing. You can also ask follow-ups like "
                "*what if it leaves 2 hours later?* or *why?*")

    def reset(self) -> None:
        self.slots: dict = {"carrier": None, "origin": None, "dest": None,
                            "date": None, "time": None, "flight_num": None}
        self.last_answer: dict | None = None

    def reply(self, text: str) -> str:
        text = text.strip()
        lower = text.lower()
        if not text:
            return "Tell me about your flight, for example *United from Denver to Seattle on Friday at 6 pm*."

        if re.search(r"\b(start over|reset|new flight|clear)\b", lower):
            self.reset()
            return "Okay, starting over. " + self._ask_next()

        if re.search(r"\b(help|what can you do)\b", lower):
            return self._help()

        if re.search(r"\b(why|explain|reason|reasons)\b", lower):
            return self._explain()

        updates = self._extract(text)
        shift = find_hour_shift(text)
        if shift is not None and self.slots["time"] is not None:
            base = datetime.combine(date.today(), self.slots["time"])
            updates["time"] = (base + timedelta(minutes=shift)).time()

        if not updates:
            if re.search(r"^(hi|hello|hey|hiya|good (morning|afternoon|evening))\b", lower):
                return self.greeting()
            return ("Sorry, I didn't catch a flight detail in that. " + self._ask_next())

        for key, value in updates.items():
            if key == "date" and not (MIN_DATE <= value <= MAX_DATE):
                return "I can check flights between 2023 and 2030 only. Which date should I use?"
            self.slots[key] = value
        if self.slots["origin"] and self.slots["origin"] == self.slots["dest"]:
            self.slots["dest"] = None
            return ("The origin and destination can't be the same airport. "
                    "Where are you flying **to**?")

        if self._missing():
            return self._summary_so_far() + self._ask_next()
        return self._answer()

    # ---- helpers -------------------------------------------------------
    def _extract(self, text: str) -> dict:
        found: dict = {}
        carrier = find_airline(text, self.carriers)
        flight_num, flight_carrier = find_flight_number(text, self.carriers)
        if flight_carrier:
            carrier = flight_carrier
        if carrier:
            found["carrier"] = carrier
        if flight_num is not None:
            found["flight_num"] = flight_num
        found.update(assign_airports(text, self.airports))
        day = find_date(text, self.today_fn())
        if day:
            found["date"] = day
        clock = find_time(text)
        if clock:
            found["time"] = clock
        return found

    def _missing(self) -> list[str]:
        return [key for key in REQUIRED if self.slots[key] is None]

    def _ask_next(self) -> str:
        missing = self._missing()
        if not missing:
            return ""
        return QUESTIONS[missing[0]]

    def _summary_so_far(self) -> str:
        parts = []
        s = self.slots
        if s["carrier"]:
            name = self.airline_names.get(s["carrier"], s["carrier"])
            parts.append(f"{name} {s['carrier']}{s['flight_num']}" if s["flight_num"] else name)
        if s["origin"] or s["dest"]:
            parts.append(f"{s['origin'] or '?'} → {s['dest'] or '?'}")
        if s["date"]:
            parts.append(s["date"].strftime("%a %d %b %Y"))
        if s["time"]:
            parts.append(s["time"].strftime("%H:%M"))
        return ("Got it: " + ", ".join(parts) + ".\n\n") if parts else ""

    def _answer(self) -> str:
        s = self.slots
        result = self.predict({
            "carrier": s["carrier"], "origin": s["origin"], "dest": s["dest"],
            "flight_num": s["flight_num"], "date": s["date"], "time": s["time"],
        })
        self.last_answer = result
        name = self.airline_names.get(s["carrier"], s["carrier"])
        number = f" {s['carrier']}{s['flight_num']}" if s["flight_num"] else ""
        when = f"{s['date'].strftime('%a %d %b %Y')} at {s['time'].strftime('%H:%M')}"
        pct = round(result["probability"] * 100)
        return (
            f"**{pct}% chance** ({result['label']}) that {name}{number} from "
            f"**{s['origin']} → {s['dest']}** on {when} arrives 15+ minutes late.\n\n"
            f"Ask me to change anything, for example *what if it leaves 2 hours later?*, "
            f"*try friday*, or *why?* to see the reasons."
        )

    def _explain(self) -> str:
        if not self.last_answer:
            return "Tell me about a flight first, then I can explain the estimate."
        r = self.last_answer
        s = self.slots
        return (
            f"Here is what went into the estimate for **{s['origin']} → {s['dest']}**:\n\n"
            f"- Typical late rate for {self.airline_names.get(s['carrier'], s['carrier'])}: {r['carrier_rate']:.0%}\n"
            f"- Typical late rate for this route: {r['route_rate']:.0%}\n"
            f"- Typical late rate leaving {s['origin']} around {s['time'].hour:02d}:00: {r['origin_hour_rate']:.0%}\n"
            f"- Weather at {s['origin']} ({r['weather_source']}): {r['weather_text']}\n\n"
            "The model combines these with the date, time and distance. "
            "Delays are noisy, so treat the result as a likelihood, not a guarantee."
        )

    def _help(self) -> str:
        return ("I can check how likely a flight is to arrive 15+ minutes late.\n\n"
                "- Give me the airline, the departure and arrival airports, the date and the time.\n"
                "- Flight numbers are optional (for example *AA 1234*).\n"
                "- Follow-ups: *what if it leaves 2 hours later?*, *try tomorrow*, *why?*, *start over*.")
