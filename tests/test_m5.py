import unittest

from nod.config import load_config
from nod.voice import Command, apply_command, format_text, parse_command, parse_numbers
from tests.fakes import FakeActions


class NumberTests(unittest.TestCase):
    def test_words(self):
        cases = {("twelve", "two"): [12, 2], ("twenty", "one", "two"): [21, 2], ("three", "sixteen"): [3, 16],
                 ("one", "hundred", "nineteen"): [119], ("eight", "twenty", "eight"): [8, 28],
                 ("one", "hundred", "twenty", "one"): [121], ("two",): [2], ("12", "2"): [12, 2],
                 ("forty",): [40], ("one", "hundred", "and", "fifty"): [150]}
        for words, expected in cases.items():
            self.assertEqual(parse_numbers(list(words)), expected, words)

    def test_non_numbers(self):
        self.assertIsNone(parse_numbers(["twelve", "banana"]))
        self.assertIsNone(parse_numbers(["hundred"]))


class FormatTests(unittest.TestCase):
    def check(self, said, expected):
        self.assertEqual(format_text(said), expected, said)

    def test_the_example_from_the_plan(self):
        self.check("romans twelve two", "Romans 12:2")

    def test_scripture_variants(self):
        self.check("john three sixteen", "John 3:16")
        self.check("psalm twenty three", "Psalm 23")
        self.check("psalms twenty three one", "Psalm 23:1")
        self.check("first corinthians thirteen four", "1 Corinthians 13:4")
        self.check("second timothy three sixteen", "2 Timothy 3:16")
        self.check("one john four eight", "1 John 4:8")
        self.check("song of solomon two four", "Song of Solomon 2:4")
        self.check("revelation twenty one four", "Revelation 21:4")
        self.check("genesis one", "Genesis 1")
        self.check("psalm one hundred nineteen", "Psalm 119")

    def test_chapter_and_verse_words(self):
        self.check("romans chapter twelve verse two", "Romans 12:2")
        self.check("john chapter three verse sixteen", "John 3:16")

    def test_ranges(self):
        self.check("romans twelve one through five", "Romans 12:1-5")

    def test_inside_a_sentence_and_punctuation(self):
        self.check("please read romans twelve two today", "please read Romans 12:2 today")
        self.check("see John three sixteen.", "see John 3:16.")
        self.check("Romans twelve two,", "Romans 12:2,")

    def test_already_formatted_by_whisper_is_left_alone(self):
        self.check("Romans 12:2", "Romans 12:2")
        self.check("romans 12 2", "Romans 12:2")

    def test_normal_words_untouched(self):
        self.check("my job is fine", "my job is fine")
        self.check("I mark the numbers", "I mark the numbers")
        self.check("hello world", "hello world")
        self.check("john said hello", "john said hello")     # no number after the name: not a reference
        self.check("I have two cats", "I have two cats")


class CommandTests(unittest.TestCase):
    def test_basic_commands(self):
        self.assertEqual(parse_command("Click."), Command("click"))
        self.assertEqual(parse_command("pause"), Command("pause"))
        self.assertEqual(parse_command("Resume!"), Command("resume"))
        self.assertEqual(parse_command("scroll down"), Command("scroll_down", 1.0))
        self.assertEqual(parse_command("Scroll up."), Command("scroll_up", 1.0))
        self.assertEqual(parse_command("scroll down a lot"), Command("scroll_down", 3.0))
        self.assertEqual(parse_command("scroll up a little"), Command("scroll_up", 0.4))

    def test_filler_words_ignored(self):
        self.assertEqual(parse_command("please click"), Command("click"))
        self.assertEqual(parse_command("Nod, scroll down"), Command("scroll_down", 1.0))

    def test_type_command_formats_and_drops_final_period(self):
        self.assertEqual(parse_command("Type romans twelve two."), Command("type", "Romans 12:2"))
        self.assertEqual(parse_command("type hello there"), Command("type", "hello there"))
        self.assertEqual(parse_command("write john three sixteen"), Command("type", "John 3:16"))

    def test_unclear_speech_is_ignored(self):
        for said in ["", "   ", "Thank you.", "banana", "type", "scroll sideways", "click the red button please"]:
            self.assertIsNone(parse_command(said), said)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.v = load_config()["voice"]
        self.fake = FakeActions()

    def test_each_command_does_its_thing(self):
        apply_command(Command("click"), self.fake, self.v)
        apply_command(Command("scroll_down", 1.0), self.fake, self.v)
        apply_command(Command("scroll_up", 3.0), self.fake, self.v)
        apply_command(Command("type", "Romans 12:2"), self.fake, self.v)
        self.assertEqual(self.fake.clicks, 1)
        self.assertEqual(self.fake.scrolls, [-self.v["scroll_clicks"], 3 * self.v["scroll_clicks"]])
        self.assertEqual(self.fake.typed, ["Romans 12:2"])

    def test_pause_and_resume(self):
        apply_command(Command("pause"), self.fake, self.v)
        self.assertTrue(self.fake.paused)
        apply_command(Command("resume"), self.fake, self.v)
        self.assertFalse(self.fake.paused)

    def test_paused_blocks_click_scroll_and_typing(self):
        self.fake.paused = True
        apply_command(Command("click"), self.fake, self.v)
        apply_command(Command("scroll_down", 1.0), self.fake, self.v)
        apply_command(Command("type", "hi"), self.fake, self.v)
        self.assertEqual((self.fake.clicks, self.fake.scrolls, self.fake.typed), (0, [], []))

    def test_voice_is_off_by_default(self):
        self.assertFalse(self.v["enabled"])


class VoiceInputSafetyTests(unittest.TestCase):
    def test_transcribe_with_no_audio_does_not_load_model_or_crash(self):
        from nod.voice_input import VoiceInput
        heard = []
        vi = VoiceInput(load_config()["voice"], heard.append)
        vi._transcribe([])
        self.assertEqual(heard, [])
        self.assertIsNone(vi._model)


if __name__ == "__main__":
    unittest.main()
