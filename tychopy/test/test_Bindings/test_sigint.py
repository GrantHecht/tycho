"""import tychopy must not install a C-level SIGINT handler.

Pre-fix, NB_MODULE installed a handler that fmt::print'ed and exit()'d,
breaking KeyboardInterrupt, Jupyter interrupts, finally blocks, and
atexit hooks in any host program. (CODEBASE_REVIEW 1.1)
"""

import os
import subprocess
import sys
import unittest


class TestSigintHandling(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX signal semantics")
    def test_import_does_not_hijack_sigint(self):
        code = (
            "import tychopy\nimport os, signal\nos.kill(os.getpid(), signal.SIGINT)\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotIn("Interrupt signal", r.stdout + r.stderr)
        self.assertIn("KeyboardInterrupt", r.stderr)


if __name__ == "__main__":
    unittest.main(exit=False)
