# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Determine facts about the environment."""

import os
import platform
import sys

# Operating systems.
WINDOWS = sys.platform == "win32"
LINUX = sys.platform == "linux2"

# Python implementations.
PYPY = (platform.python_implementation() == 'PyPy')
if PYPY:
    PYPYVERSION = sys.pypy_version_info

JYTHON = (platform.python_implementation() == 'Jython')
IRONPYTHON = (platform.python_implementation() == 'IronPython')

# Python versions.
PYVERSION = sys.version_info
PY2 = PYVERSION < (3, 0)
PY3 = PYVERSION >= (3, 0)

# Python behavior
class PYBEHAVIOR(object):
    """Flags indicating this Python's behavior."""

    # Is "if __debug__" optimized away?
    optimize_if_debug = (not PYPY)

    # Is "if not __debug__" optimized away?
    optimize_if_not_debug = (not PYPY) and (PYVERSION >= (3, 7, 0, 'alpha', 4))

    # Is "if not __debug__" optimized away even better?
    optimize_if_not_debug2 = (not PYPY) and (PYVERSION >= (3, 8, 0, 'beta', 1))

    # Do we have yield-from?
    yield_from = (PYVERSION >= (3, 3))

    # Do we have PEP 420 namespace packages?
    namespaces_pep420 = (PYVERSION >= (3, 3))

    # Do .pyc files have the source file size recorded in them?
    size_in_pyc = (PYVERSION >= (3, 3))

    # Do we have async and await syntax?
    async_syntax = (PYVERSION >= (3, 5))

    # PEP 448 defined additional unpacking generalizations
    unpackings_pep448 = (PYVERSION >= (3, 5))

    # Can co_lnotab have negative deltas?
    negative_lnotab = (PYVERSION >= (3, 6))

    # Do .pyc files conform to PEP 552? Hash-based pyc's.
    hashed_pyc_pep552 = (PYVERSION >= (3, 7, 0, 'alpha', 4))

    # Python 3.7.0b3 changed the behavior of the sys.path[0] entry for -m. It
    # used to be an empty string (meaning the current directory). It changed
    # to be the actual path to the current directory, so that os.chdir wouldn't
    # affect the outcome.
    actual_syspath0_dash_m = (PYVERSION >= (3, 7, 0, 'beta', 3))

    # When a break/continue/return statement in a try block jumps to a finally
    # block, does the finally block do the break/continue/return (pre-3.8), or
    # does the finally jump back to the break/continue/return (3.8) to do the
    # work?
    finally_jumps_back = (PYVERSION >= (3, 8))

    # When a function is decorated, does the trace function get called for the
    # @-line and also the def-line (new behavior in 3.8)? Or just the @-line
    # (old behavior)?
    trace_decorated_def = (PYVERSION >= (3, 8))

    # Are while-true loops optimized into absolute jumps with no loop setup?
    nix_while_true = (PYVERSION >= (3, 8))

# Coverage.py specifics.

# Are we using the C-implemented trace function?
C_TRACER = os.getenv('COVERAGE_TEST_TRACER', 'c') == 'c'

# Are we coverage-measuring ourselves?
METACOV = os.getenv('COVERAGE_COVERAGE', '') != ''

# Are we running our test suite?
# Even when running tests, you can use COVERAGE_TESTING=0 to disable the
# test-specific behavior like contracts.
TESTING = os.getenv('COVERAGE_TESTING', '') == 'True'
