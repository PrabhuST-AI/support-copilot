"""validate.py -- Validation script that asserts reference checkpoints from tasks 3 and 4
and prints exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""

import sys
from store_and_validate import run_validations

if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    run_validations(verbose=verbose)
    print("ALL CHECKPOINTS PASSED")
