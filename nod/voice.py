"""Voice commands: turn what you said (text) into an action. No audio code here.

Commands: "scroll up", "scroll down" (add "a lot" / "a little"), "click",
          "pause", "resume", "type <anything>".
Smart formatting for typing: "romans twelve two" -> "Romans 12:2".
"""
import re
from dataclasses import dataclass

# ---------- number words ----------
UNITS = {w: i for i, w in enumerate("one two three four five six seven eight nine".split(), 1)}
TEENS = {w: 10 + i for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}


def parse_numbers(tokens):
    """['twelve','two'] -> [12, 2];  ['twenty','one','two'] -> [21, 2].  None if any token isn't a number."""
    nums, cur, last, hundred = [], None, None, False
    for tok in tokens:
        if tok == "and":
            continue
        if tok == "hundred":
            if cur is None or not 1 <= cur <= 9:
                return None
            cur, hundred, last = cur * 100, True, "hundred"
            continue
        if tok.isdigit():
            v, kind = int(tok), "digit"
        elif tok in UNITS:
            v, kind = UNITS[tok], "unit"
        elif tok in TEENS:
            v, kind = TEENS[tok], "teen"
        elif tok in TENS:
            v, kind = TENS[tok], "tens"
        else:
            return None
        if cur is None:
            cur, last = v, kind
        elif hundred and cur % 100 == 0 and kind in ("unit", "teen", "tens"):
            cur += v                                   # "one hundred nineteen" -> 119
            hundred, last = (False, kind) if kind != "tens" else (True, "tens")
        elif last == "tens" and kind == "unit" and cur % 10 == 0:
            cur += v                                   # "twenty one" -> 21
            hundred, last = False, "unit"
        else:
            nums.append(cur)                           # a new number starts: "twelve" | "two"
            cur, last, hundred = v, kind, False
    if cur is not None:
        nums.append(cur)
    return nums


# ---------- books of the Bible ----------
_NUMBERED = {"samuel": 2, "kings": 2, "chronicles": 2, "corinthians": 2, "thessalonians": 2,
             "timothy": 2, "peter": 2, "john": 3}
_PLAIN = ("genesis exodus leviticus numbers deuteronomy joshua judges ruth ezra nehemiah esther job "
          "proverbs ecclesiastes isaiah jeremiah lamentations ezekiel daniel hosea joel amos obadiah jonah "
          "micah nahum habakkuk zephaniah haggai zechariah malachi matthew mark luke acts romans galatians "
          "ephesians philippians colossians titus philemon hebrews james jude").split()
_PREFIX = {"first": 1, "1": 1, "second": 2, "2": 2, "third": 3, "3": 3}
_ONE_TWO_THREE = {"one": 1, "two": 2, "three": 3}   # "one john" - only accepted in front of a numbered book

# spoken tokens (tuple) -> canonical name
BOOKS = {(b,): b.capitalize() for b in _PLAIN}
BOOKS.update({("psalm",): "Psalm", ("psalms",): "Psalm", ("revelation",): "Revelation",
              ("revelations",): "Revelation", ("song", "of", "solomon"): "Song of Solomon",
              ("song", "of", "songs"): "Song of Solomon"})
BOOKS[("john",)] = "John"   # the Gospel; "first/second/third john" are handled in _match_book
_BOOK_KEYS_LONGEST_FIRST = sorted(BOOKS, key=len, reverse=True)

_TRAIL = re.compile(r"[.,;:!?)\]\"']+$")


def _key(word):
    """Lowercase word with outer punctuation stripped, but keeping things like '12:2' intact."""
    return re.sub(r"^[(\[\"']+|[.,;!?)\]\"']+$", "", word).lower()


def _trail(word):
    m = _TRAIL.search(word)
    return m.group(0) if m else ""


def _match_book(keys, i):
    """If a book name starts at keys[i], return (canonical_name, tokens_used), else None."""
    # numbered books: "first john", "1 john", "one john"
    if i + 1 < len(keys) and keys[i + 1] in _NUMBERED:
        n = _PREFIX.get(keys[i]) or _ONE_TWO_THREE.get(keys[i])
        if n and n <= _NUMBERED[keys[i + 1]]:
            return f"{n} {keys[i + 1].capitalize()}", 2
    for book in _BOOK_KEYS_LONGEST_FIRST:
        if tuple(keys[i:i + len(book)]) == book:
            return BOOKS[book], len(book)
    return None


def _numeric_run(keys, j):
    """Collect following tokens that are numbers or 'chapter'/'verse'. Returns (end_index, numbers, saw_verse_word)."""
    k, chunk, split_at, nums_before = j, [], None, 0
    while k < len(keys):
        w = keys[k]
        if w in ("chapter", "chapters"):
            k += 1
            continue
        if w in ("verse", "verses"):
            parsed = parse_numbers(chunk) if chunk else []
            if parsed is None or not parsed:
                break
            nums_before, split_at = len(parsed), len(chunk)
            k += 1
            continue
        if w == "and" and (k + 1 >= len(keys) or parse_numbers([keys[k + 1]]) is None):
            break
        if parse_numbers([w]) is None and w != "and" and w != "hundred":
            break
        chunk.append(w)
        k += 1
    if not chunk:
        return j, None, False
    if split_at is not None:
        a, b = parse_numbers(chunk[:split_at]), parse_numbers(chunk[split_at:])
        nums = (a or []) + (b or [])
    else:
        nums = parse_numbers(chunk)
    return k, nums, split_at is not None


def format_text(text):
    """'type romans twelve two' style text -> 'Romans 12:2'. Everything else passes through unchanged."""
    words = text.split()
    keys = [_key(w) for w in words]
    out, i = [], 0
    while i < len(words):
        book = _match_book(keys, i)
        if book:
            name, used = book
            end, nums, _ = _numeric_run(keys, i + used)
            if nums and 1 <= len(nums) <= 2:
                ref = f"{name} {nums[0]}" + (f":{nums[1]}" if len(nums) == 2 else "")
                # "through": romans twelve one through five -> Romans 12:1-5
                if len(nums) == 2 and end < len(keys) and keys[end] in ("through", "thru"):
                    e2, more, _ = _numeric_run(keys, end + 1)
                    if more and len(more) == 1:
                        ref += f"-{more[0]}"
                        end = e2
                out.append(ref + _trail(words[end - 1]))
                i = end
                continue
        out.append(words[i])
        i += 1
    return " ".join(out)


# ---------- commands ----------
@dataclass
class Command:
    kind: str          # scroll_up, scroll_down, click, pause, resume, type
    arg: object = None  # scroll multiplier, or text to type


_FILLER = {"please", "nod", "hey", "okay", "ok"}
_TYPE_WORDS = {"type", "write", "dictate"}


def parse_command(text):
    """Returns a Command, or None if we didn't understand (we ignore anything unclear)."""
    raw = text.strip()
    words = raw.split()
    while words and _key(words[0]) in _FILLER:
        words.pop(0)
    if not words:
        return None
    first = _key(words[0])

    if first in _TYPE_WORDS and len(words) > 1:
        body = " ".join(words[1:])
        body = re.sub(r"[.,]$", "", body)       # speech-to-text adds a final period; don't type it
        return Command("type", format_text(body))

    plain = " ".join(re.sub(r"[^\w\s]", " ", " ".join(words).lower()).split())
    if plain in ("click", "left click", "click it", "click here"):
        return Command("click")
    if plain in ("pause", "stop", "pause nod", "stop listening", "freeze"):
        return Command("pause")
    if plain in ("resume", "continue", "start", "unpause", "resume nod"):
        return Command("resume")
    m = re.fullmatch(r"scroll (up|down)( a lot| more| a little| a bit| little)?", plain)
    if m:
        amount = {" a lot": 3.0, " more": 2.0, " a little": 0.4, " a bit": 0.4, " little": 0.4}.get(m.group(2), 1.0)
        return Command("scroll_" + m.group(1), amount)
    return None


def apply_command(cmd, actions, vcfg):
    """Do it. `actions` is nod.actions (or a fake). Every action already respects pause."""
    if cmd.kind == "click":
        actions.click()
    elif cmd.kind in ("scroll_up", "scroll_down"):
        n = max(1, round(vcfg["scroll_clicks"] * cmd.arg))
        actions.scroll(n if cmd.kind == "scroll_up" else -n)
    elif cmd.kind == "pause":
        actions.set_paused(True)
    elif cmd.kind == "resume":
        actions.set_paused(False)
    elif cmd.kind == "type":
        actions.type_text(cmd.arg)
