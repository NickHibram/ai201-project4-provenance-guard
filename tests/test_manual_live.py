import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.manual_live import speech_cases


class OptionalSpeechCasesTests(unittest.TestCase):
    def test_loads_only_speech_excerpts_supplied_as_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            mlk_path = Path(directory) / 'mlk.txt'
            malcolm_path = Path(directory) / 'malcolm.txt'
            mlk_path.write_text('MLK passage.', encoding='utf-8')
            malcolm_path.write_text('Malcolm passage.', encoding='utf-8')
            with patch.dict(os.environ, {'MLK_SPEECH_PATH': str(mlk_path),
                                      'MALCOLM_SPEECH_PATH': str(malcolm_path)}):
                self.assertEqual(speech_cases(), [
                    ('human_mlk_dream', 'MLK passage.'),
                    ('human_malcolm_ballot_bullet', 'Malcolm passage.'),
                ])


if __name__ == '__main__':
    unittest.main()
