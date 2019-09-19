# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Core control stuff for coverage.py."""

import atexit
import os
import os.path
import platform
import sys
import time

from coverage import env
from coverage.annotate import AnnotateReporter
from coverage.backward import string_class, iitems
from coverage.collector import Collector, CTracer
from coverage.config import read_coverage_config
from coverage.context import should_start_context_test_function, combine_context_switchers
from coverage.data import CoverageData, combine_parallel_data
from coverage.debug import DebugControl, write_formatted_info
from coverage.disposition import disposition_debug_msg
from coverage.files import PathAliases, set_relative_directory, abs_file
from coverage.html import HtmlReporter
from coverage.inorout import InOrOut
from coverage.misc import CoverageException, bool_or_none, join_regex
from coverage.misc import ensure_dir_for_file, file_be_gone, isolate_module
from coverage.plugin import FileReporter
from coverage.plugin_support import Plugins
from coverage.python import PythonFileReporter
from coverage.results import Analysis, Numbers
from coverage.summary import SummaryReporter
from coverage.xmlreport import XmlReporter

try:
    from coverage.multiproc import patch_multiprocessing
except ImportError:                                         # pragma: only jython
    # Jython has no multiprocessing module.
    patch_multiprocessing = None

os = isolate_module(os)


class Coverage(object):
    """Programmatic access to coverage.py.

    To use::

        from coverage import Coverage

        cov = Coverage()
        cov.start()
        #.. call your code ..
        cov.stop()
        cov.html_report(directory='covhtml')

    """

    # The stack of started Coverage instances.
    _instances = []

    @classmethod
    def current(cls):
        """Get the latest started `Coverage` instance, if any.

        Returns: a `Coverage` instance, or None.

        .. versionadded:: 5.0

        """
        if cls._instances:
            return cls._instances[-1]
        else:
            return None

    def __init__(
        self, data_file=None, data_suffix=None, cover_pylib=None,
        auto_data=False, timid=None, branch=None, config_file=True,
        source=None, omit=None, include=None, debug=None,
        concurrency=None, check_preimported=False, context=None,
    ):
        """
        `data_file` is the base name of the data file to use, defaulting to
        ".coverage".  `data_suffix` is appended (with a dot) to `data_file` to
        create the final file name.  If `data_suffix` is simply True, then a
        suffix is created with the machine and process identity included.

        `cover_pylib` is a boolean determining whether Python code installed
        with the Python interpreter is measured.  This includes the Python
        standard library and any packages installed with the interpreter.

        If `auto_data` is true, then any existing data file will be read when
        coverage measurement starts, and data will be saved automatically when
        measurement stops.

        If `timid` is true, then a slower and simpler trace function will be
        used.  This is important for some environments where manipulation of
        tracing functions breaks the faster trace function.

        If `branch` is true, then branch coverage will be measured in addition
        to the usual statement coverage.

        `config_file` determines what configuration file to read:

            * If it is ".coveragerc", it is interpreted as if it were True,
              for backward compatibility.

            * If it is a string, it is the name of the file to read.  If the
              file can't be read, it is an error.

            * If it is True, then a few standard files names are tried
              (".coveragerc", "setup.cfg", "tox.ini").  It is not an error for
              these files to not be found.

            * If it is False, then no configuration file is read.

        `source` is a list of file paths or package names.  Only code located
        in the trees indicated by the file paths or package names will be
        measured.

        `include` and `omit` are lists of file name patterns. Files that match
        `include` will be measured, files that match `omit` will not.  Each
        will also accept a single string argument.

        `debug` is a list of strings indicating what debugging information is
        desired.

        `concurrency` is a string indicating the concurrency library being used
        in the measured code.  Without this, coverage.py will get incorrect
        results if these libraries are in use.  Valid strings are "greenlet",
        "eventlet", "gevent", "multiprocessing", or "thread" (the default).
        This can also be a list of these strings.

        If `check_preimported` is true, then when coverage is started, the
        already-imported files will be checked to see if they should be measured
        by coverage.  Importing measured files before coverage is started can
        mean that code is missed.

        `context` is a string to use as the context label for collected data.

        .. versionadded:: 4.0
            The `concurrency` parameter.

        .. versionadded:: 4.2
            The `concurrency` parameter can now be a list of strings.

        .. versionadded:: 5.0
            The `check_preimported` and `context` parameters.

        """
        # Build our configuration from a number of sources.
        self.config = read_coverage_config(
            config_file=config_file,
            data_file=data_file, cover_pylib=cover_pylib, timid=timid,
            branch=branch, parallel=bool_or_none(data_suffix),
            source=source, run_omit=omit, run_include=include, debug=debug,
            report_omit=omit, report_include=include,
            concurrency=concurrency, context=context,
            )

        # This is injectable by tests.
        self._debug_file = None

        self._auto_load = self._auto_save = auto_data
        self._data_suffix_specified = data_suffix

        # Is it ok for no data to be collected?
        self._warn_no_data = True
        self._warn_unimported_source = True
        self._warn_preimported_source = check_preimported

        # A record of all the warnings that have been issued.
        self._warnings = []

        # Other instance attributes, set later.
        self._data = self._collector = None
        self._plugins = None
        self._inorout = None
        self._inorout_class = InOrOut
        self._data_suffix = self._run_suffix = None
        self._exclude_re = None
        self._debug = None

        # State machine variables:
        # Have we initialized everything?
        self._inited = False
        self._inited_for_start = False
        # Have we started collecting and not stopped it?
        self._started = False
        # Have we written --debug output?
        self._wrote_debug = False

        # If we have sub-process measurement happening automatically, then we
        # want any explicit creation of a Coverage object to mean, this process
        # is already coverage-aware, so don't auto-measure it.  By now, the
        # auto-creation of a Coverage object has already happened.  But we can
        # find it and tell it not to save its data.
        if not env.METACOV:
            _prevent_sub_process_measurement()

    def _init(self):
        """Set all the initial state.

        This is called by the public methods to initialize state. This lets us
        construct a :class:`Coverage` object, then tweak its state before this
        function is called.

        """
        if self._inited:
            return

        self._inited = True

        # Create and configure the debugging controller. COVERAGE_DEBUG_FILE
        # is an environment variable, the name of a file to append debug logs
        # to.
        self._debug = DebugControl(self.config.debug, self._debug_file)

        # _exclude_re is a dict that maps exclusion list names to compiled regexes.
        self._exclude_re = {}

        set_relative_directory()

        # Load plugins
        self._plugins = Plugins.load_plugins(self.config.plugins, self.config, self._debug)

        # Run configuring plugins.
        for plugin in self._plugins.configurers:
            # We need an object with set_option and get_option. Either self or
            # self.config will do. Choosing randomly stops people from doing
            # other things with those objects, against the public API.  Yes,
            # this is a bit childish. :)
            plugin.configure([self, self.config][int(time.time()) % 2])

    def _post_init(self):
        """Stuff to do after everything is initialized."""
        if not self._wrote_debug:
            self._wrote_debug = True
            self._write_startup_debug()

    def _write_startup_debug(self):
        """Write out debug info at startup if needed."""
        wrote_any = False
        with self._debug.without_callers():
            if self._debug.should('config'):
                config_info = sorted(self.config.__dict__.items())
                config_info = [(k, v) for k, v in config_info if not k.startswith('_')]
                write_formatted_info(self._debug, "config", config_info)
                wrote_any = True

            if self._debug.should('sys'):
                write_formatted_info(self._debug, "sys", self.sys_info())
                for plugin in self._plugins:
                    header = "sys: " + plugin._coverage_plugin_name
                    info = plugin.sys_info()
                    write_formatted_info(self._debug, header, info)
                wrote_any = True

        if wrote_any:
            write_formatted_info(self._debug, "end", ())

    def _should_trace(self, filename, frame):
        """Decide whether to trace execution in `filename`.

        Calls `_should_trace_internal`, and returns the FileDisposition.

        """
        disp = self._inorout.should_trace(filename, frame)
        if self._debug.should('trace'):
            self._debug.write(disposition_debug_msg(disp))
        return disp

    def _check_include_omit_etc(self, filename, frame):
        """Check a file name against the include/omit/etc, rules, verbosely.

        Returns a boolean: True if the file should be traced, False if not.

        """
        reason = self._inorout.check_include_omit_etc(filename, frame)
        if self._debug.should('trace'):
            if not reason:
                msg = "Including %r" % (filename,)
            else:
                msg = "Not including %r: %s" % (filename, reason)
            self._debug.write(msg)

        return not reason

    def _warn(self, msg, slug=None):
        """Use `msg` as a warning.

        For warning suppression, use `slug` as the shorthand.
        """
        if slug in self.config.disable_warnings:
            # Don't issue the warning
            return

        self._warnings.append(msg)
        if slug:
            msg = "%s (%s)" % (msg, slug)
        if self._debug.should('pid'):
            msg = "[%d] %s" % (os.getpid(), msg)
        sys.stderr.write("Coverage.py warning: %s\n" % msg)

    def get_option(self, option_name):
        """Get an option from the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with `"run:branch"`.

        Returns the value of the option.

        .. versionadded:: 4.0

        """
        return self.config.get_option(option_name)

    def set_option(self, option_name, value):
        """Set an option in the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with ``"run:branch"``.

        `value` is the new value for the option.  This should be an
        appropriate Python value.  For example, use True for booleans, not the
        string ``"True"``.

        As an example, calling::

            cov.set_option("run:branch", True)

        has the same effect as this configuration file::

            [run]
            branch = True

        .. versionadded:: 4.0

        """
        self.config.set_option(option_name, value)

    def load(self):
        """Load previously-collected coverage data from the data file."""
        self._init()
        if self._collector:
            self._collector.reset()
        should_skip = self.config.parallel and not os.path.exists(self.config.data_file)
        if not should_skip:
            self._init_data(suffix=None)
        self._post_init()
        if not should_skip:
            self._data.read()

    def _init_for_start(self):
        """Initialization for start()"""
        # Construct the collector.
        concurrency = self.config.concurrency or []
        if "multiprocessing" in concurrency:
            if not patch_multiprocessing:
                raise CoverageException(                    # pragma: only jython
                    "multiprocessing is not supported on this Python"
                )
            patch_multiprocessing(rcfile=self.config.config_file)
            # Multi-processing uses parallel for the subprocesses, so also use
            # it for the main process.
            self.config.parallel = True

        dycon = self.config.dynamic_context
        if not dycon or dycon == "none":
            context_switchers = []
        elif dycon == "test_function":
            context_switchers = [should_start_context_test_function]
        else:
            raise CoverageException(
                "Don't understand dynamic_context setting: {!r}".format(dycon)
            )

        context_switchers.extend(
            plugin.dynamic_context for plugin in self._plugins.context_switchers
        )

        should_start_context = combine_context_switchers(context_switchers)

        self._collector = Collector(
            should_trace=self._should_trace,
            check_include=self._check_include_omit_etc,
            should_start_context=should_start_context,
            timid=self.config.timid,
            branch=self.config.branch,
            warn=self._warn,
            concurrency=concurrency,
            )

        suffix = self._data_suffix_specified
        if suffix or self.config.parallel:
            if not isinstance(suffix, string_class):
                # if data_suffix=True, use .machinename.pid.random
                suffix = True
        else:
            suffix = None

        self._init_data(suffix)

        self._collector.use_data(self._data, self.config.context)

        # Early warning if we aren't going to be able to support plugins.
        if self._plugins.file_tracers and not self._collector.supports_plugins:
            self._warn(
                "Plugin file tracers (%s) aren't supported with %s" % (
                    ", ".join(
                        plugin._coverage_plugin_name
                            for plugin in self._plugins.file_tracers
                        ),
                    self._collector.tracer_name(),
                    )
                )
            for plugin in self._plugins.file_tracers:
                plugin._coverage_enabled = False

        # Create the file classifying substructure.
        self._inorout = self._inorout_class(warn=self._warn)
        self._inorout.configure(self.config)
        self._inorout.plugins = self._plugins
        self._inorout.disp_class = self._collector.file_disposition_class

        atexit.register(self._atexit)

    def _init_data(self, suffix):
        """Create a data file if we don't have one yet."""
        if self._data is None:
            # Create the data file.  We do this at construction time so that the
            # data file will be written into the directory where the process
            # started rather than wherever the process eventually chdir'd to.
            ensure_dir_for_file(self.config.data_file)
            self._data = CoverageData(
                basename=self.config.data_file,
                suffix=suffix,
                warn=self._warn,
                debug=self._debug,
            )

    def start(self):
        """Start measuring code coverage.

        Coverage measurement only occurs in functions called after
        :meth:`start` is invoked.  Statements in the same scope as
        :meth:`start` won't be measured.

        Once you invoke :meth:`start`, you must also call :meth:`stop`
        eventually, or your process might not shut down cleanly.

        """
        self._init()
        if not self._inited_for_start:
            self._inited_for_start = True
            self._init_for_start()
        self._post_init()

        # Issue warnings for possible problems.
        self._inorout.warn_conflicting_settings()

        # See if we think some code that would eventually be measured has
        # already been imported.
        if self._warn_preimported_source:
            self._inorout.warn_already_imported_files()

        if self._auto_load:
            self.load()

        self._collector.start()
        self._started = True
        self._instances.append(self)

    def stop(self):
        """Stop measuring code coverage."""
        if self._instances:
            if self._instances[-1] is self:
                self._instances.pop()
        if self._started:
            self._collector.stop()
        self._started = False

    def _atexit(self):
        """Clean up on process shutdown."""
        if self._debug.should("process"):
            self._debug.write("atexit: {0!r}".format(self))
        if self._started:
            self.stop()
        if self._auto_save:
            self.save()

    def erase(self):
        """Erase previously-collected coverage data.

        This removes the in-memory data collected in this session as well as
        discarding the data file.

        """
        self._init()
        self._post_init()
        if self._collector:
            self._collector.reset()
        self._init_data(suffix=None)
        self._data.erase(parallel=self.config.parallel)
        self._data = None

    def switch_context(self, new_context):
        """Switch to a new dynamic context.

        `new_context` is a string to use as the context label
        for collected data.  If a :ref:`static context <static_contexts>` is in
        use, the static and dynamic context labels will be joined together with
        a pipe character.

        Coverage collection must be started already.

        .. versionadded:: 5.0
        """
        if not self._started:
            raise CoverageException(                    # pragma: only jython
                "Cannot switch context, coverage is not started"
                )
        self._collector.switch_context(new_context)

    def clear_exclude(self, which='exclude'):
        """Clear the exclude list."""
        self._init()
        setattr(self.config, which + "_list", [])
        self._exclude_regex_stale()

    def exclude(self, regex, which='exclude'):
        """Exclude source lines from execution consideration.

        A number of lists of regular expressions are maintained.  Each list
        selects lines that are treated differently during reporting.

        `which` determines which list is modified.  The "exclude" list selects
        lines that are not considered executable at all.  The "partial" list
        indicates lines with branches that are not taken.

        `regex` is a regular expression.  The regex is added to the specified
        list.  If any of the regexes in the list is found in a line, the line
        is marked for special treatment during reporting.

        """
        self._init()
        excl_list = getattr(self.config, which + "_list")
        excl_list.append(regex)
        self._exclude_regex_stale()

    def _exclude_regex_stale(self):
        """Drop all the compiled exclusion regexes, a list was modified."""
        self._exclude_re.clear()

    def _exclude_regex(self, which):
        """Return a compiled regex for the given exclusion list."""
        if which not in self._exclude_re:
            excl_list = getattr(self.config, which + "_list")
            self._exclude_re[which] = join_regex(excl_list)
        return self._exclude_re[which]

    def get_exclude_list(self, which='exclude'):
        """Return a list of excluded regex patterns.

        `which` indicates which list is desired.  See :meth:`exclude` for the
        lists that are available, and their meaning.

        """
        self._init()
        return getattr(self.config, which + "_list")

    def save(self):
        """Save the collected coverage data to the data file."""
        data = self.get_data()
        data.write()

    def combine(self, data_paths=None, strict=False):
        """Combine together a number of similarly-named coverage data files.

        All coverage data files whose name starts with `data_file` (from the
        coverage() constructor) will be read, and combined together into the
        current measurements.

        `data_paths` is a list of files or directories from which data should
        be combined. If no list is passed, then the data files from the
        directory indicated by the current data file (probably the current
        directory) will be combined.

        If `strict` is true, then it is an error to attempt to combine when
        there are no data files to combine.

        .. versionadded:: 4.0
            The `data_paths` parameter.

        .. versionadded:: 4.3
            The `strict` parameter.

        """
        self._init()
        self._init_data(suffix=None)
        self._post_init()
        self.get_data()

        aliases = None
        if self.config.paths:
            aliases = PathAliases()
            for paths in self.config.paths.values():
                result = paths[0]
                for pattern in paths[1:]:
                    aliases.add(pattern, result)

        combine_parallel_data(self._data, aliases=aliases, data_paths=data_paths, strict=strict)

    def get_data(self):
        """Get the collected data.

        Also warn about various problems collecting data.

        Returns a :class:`coverage.CoverageData`, the collected coverage data.

        .. versionadded:: 4.0

        """
        self._init()
        self._init_data(suffix=None)
        self._post_init()

        if self._collector and self._collector.flush_data():
            self._post_save_work()

        return self._data

    def _post_save_work(self):
        """After saving data, look for warnings, post-work, etc.

        Warn about things that should have happened but didn't.
        Look for unexecuted files.

        """
        # If there are still entries in the source_pkgs_unmatched list,
        # then we never encountered those packages.
        if self._warn_unimported_source:
            self._inorout.warn_unimported_source()

        # Find out if we got any data.
        if not self._data and self._warn_no_data:
            self._warn("No data was collected.", slug="no-data-collected")

        # Find files that were never executed at all.
        if self._data:
            for file_path, plugin_name in self._inorout.find_unexecuted_files():
                self._data.touch_file(file_path, plugin_name)

        if self.config.note:
            self._data.add_run_info(note=self.config.note)

    # Backward compatibility with version 1.
    def analysis(self, morf):
        """Like `analysis2` but doesn't return excluded line numbers."""
        f, s, _, m, mf = self.analysis2(morf)
        return f, s, m, mf

    def analysis2(self, morf):
        """Analyze a module.

        `morf` is a module or a file name.  It will be analyzed to determine
        its coverage statistics.  The return value is a 5-tuple:

        * The file name for the module.
        * A list of line numbers of executable statements.
        * A list of line numbers of excluded statements.
        * A list of line numbers of statements not run (missing from
          execution).
        * A readable formatted string of the missing line numbers.

        The analysis uses the source file itself and the current measured
        coverage data.

        """
        analysis = self._analyze(morf)
        return (
            analysis.filename,
            sorted(analysis.statements),
            sorted(analysis.excluded),
            sorted(analysis.missing),
            analysis.missing_formatted(),
            )

    def _analyze(self, it):
        """Analyze a single morf or code unit.

        Returns an `Analysis` object.

        """
        # All reporting comes through here, so do reporting initialization.
        self._init()
        Numbers.set_precision(self.config.precision)
        self._post_init()

        data = self.get_data()
        if not isinstance(it, FileReporter):
            it = self._get_file_reporter(it)

        return Analysis(data, it)

    def _get_file_reporter(self, morf):
        """Get a FileReporter for a module or file name."""
        plugin = None
        file_reporter = "python"

        if isinstance(morf, string_class):
            abs_morf = abs_file(morf)
            plugin_name = self._data.file_tracer(abs_morf)
            if plugin_name:
                plugin = self._plugins.get(plugin_name)

        if plugin:
            file_reporter = plugin.file_reporter(abs_morf)
            if file_reporter is None:
                raise CoverageException(
                    "Plugin %r did not provide a file reporter for %r." % (
                        plugin._coverage_plugin_name, morf
                    )
                )

        if file_reporter == "python":
            file_reporter = PythonFileReporter(morf, self)

        return file_reporter

    def _get_file_reporters(self, morfs=None):
        """Get a list of FileReporters for a list of modules or file names.

        For each module or file name in `morfs`, find a FileReporter.  Return
        the list of FileReporters.

        If `morfs` is a single module or file name, this returns a list of one
        FileReporter.  If `morfs` is empty or None, then the list of all files
        measured is used to find the FileReporters.

        """
        if not morfs:
            morfs = self._data.measured_files()

        # Be sure we have a collection.
        if not isinstance(morfs, (list, tuple, set)):
            morfs = [morfs]

        file_reporters = [self._get_file_reporter(morf) for morf in morfs]
        return file_reporters

    def report(
        self, morfs=None, show_missing=None, ignore_errors=None,
        file=None, omit=None, include=None, skip_covered=None,
        contexts=None,
    ):
        """Write a textual summary report to `file`.

        Each module in `morfs` is listed, with counts of statements, executed
        statements, missing statements, and a list of lines missed.

        If `show_missing` is true, then details of which lines or branches are
        missing will be included in the report.  If `ignore_errors` is true,
        then a failure while reporting a single file will not stop the entire
        report.

        `file` is a file-like object, suitable for writing.

        `include` is a list of file name patterns.  Files that match will be
        included in the report. Files matching `omit` will not be included in
        the report.

        If `skip_covered` is true, don't report on files with 100% coverage.

        All of the arguments default to the settings read from the
        :ref:`configuration file <config>`.

        Returns a float, the total percentage covered.

        """
        self.config.from_args(
            ignore_errors=ignore_errors, report_omit=omit, report_include=include,
            show_missing=show_missing, skip_covered=skip_covered,
            report_contexts=contexts,
            )
        reporter = SummaryReporter(self)
        return reporter.report(morfs, outfile=file)

    def annotate(
        self, morfs=None, directory=None, ignore_errors=None,
        omit=None, include=None, contexts=None,
    ):
        """Annotate a list of modules.

        Each module in `morfs` is annotated.  The source is written to a new
        file, named with a ",cover" suffix, with each line prefixed with a
        marker to indicate the coverage of the line.  Covered lines have ">",
        excluded lines have "-", and missing lines have "!".

        See :meth:`report` for other arguments.

        """
        self.config.from_args(
            ignore_errors=ignore_errors, report_omit=omit,
            report_include=include, report_contexts=contexts,
            )
        reporter = AnnotateReporter(self)
        reporter.report(morfs, directory=directory)

    def html_report(self, morfs=None, directory=None, ignore_errors=None,
                    omit=None, include=None, extra_css=None, title=None,
                    skip_covered=None, show_contexts=None, contexts=None):
        """Generate an HTML report.

        The HTML is written to `directory`.  The file "index.html" is the
        overview starting point, with links to more detailed pages for
        individual modules.

        `extra_css` is a path to a file of other CSS to apply on the page.
        It will be copied into the HTML directory.

        `title` is a text string (not HTML) to use as the title of the HTML
        report.

        See :meth:`report` for other arguments.

        Returns a float, the total percentage covered.

        .. note::
            The HTML report files are generated incrementally based on the
            source files and coverage results. If you modify the report files,
            the changes will not be considered.  You should be careful about
            changing the files in the report folder.

        """
        self.config.from_args(
            ignore_errors=ignore_errors, report_omit=omit, report_include=include,
            html_dir=directory, extra_css=extra_css, html_title=title,
            skip_covered=skip_covered, show_contexts=show_contexts, report_contexts=contexts,
            )
        reporter = HtmlReporter(self)
        return reporter.report(morfs)

    def xml_report(
        self, morfs=None, outfile=None, ignore_errors=None,
        omit=None, include=None, contexts=None,
    ):
        """Generate an XML report of coverage results.

        The report is compatible with Cobertura reports.

        Each module in `morfs` is included in the report.  `outfile` is the
        path to write the file to, "-" will write to stdout.

        See :meth:`report` for other arguments.

        Returns a float, the total percentage covered.

        """
        self.config.from_args(
            ignore_errors=ignore_errors, report_omit=omit, report_include=include,
            xml_output=outfile, report_contexts=contexts,
            )
        file_to_close = None
        delete_file = False
        if self.config.xml_output:
            if self.config.xml_output == '-':
                outfile = sys.stdout
            else:
                # Ensure that the output directory is created; done here
                # because this report pre-opens the output file.
                # HTMLReport does this using the Report plumbing because
                # its task is more complex, being multiple files.
                ensure_dir_for_file(self.config.xml_output)
                open_kwargs = {}
                if env.PY3:
                    open_kwargs['encoding'] = 'utf8'
                outfile = open(self.config.xml_output, "w", **open_kwargs)
                file_to_close = outfile
        try:
            reporter = XmlReporter(self)
            return reporter.report(morfs, outfile=outfile)
        except CoverageException:
            delete_file = True
            raise
        finally:
            if file_to_close:
                file_to_close.close()
                if delete_file:
                    file_be_gone(self.config.xml_output)

    def sys_info(self):
        """Return a list of (key, value) pairs showing internal information."""

        import coverage as covmod

        self._init()
        self._post_init()

        def plugin_info(plugins):
            """Make an entry for the sys_info from a list of plug-ins."""
            entries = []
            for plugin in plugins:
                entry = plugin._coverage_plugin_name
                if not plugin._coverage_enabled:
                    entry += " (disabled)"
                entries.append(entry)
            return entries

        info = [
            ('version', covmod.__version__),
            ('coverage', covmod.__file__),
            ('tracer', self._collector.tracer_name() if self._collector else "-none-"),
            ('CTracer', 'available' if CTracer else "unavailable"),
            ('plugins.file_tracers', plugin_info(self._plugins.file_tracers)),
            ('plugins.configurers', plugin_info(self._plugins.configurers)),
            ('configs_attempted', self.config.attempted_config_files),
            ('configs_read', self.config.config_files_read),
            ('config_file', self.config.config_file),
            ('config_contents',
                repr(self.config._config_contents)
                if self.config._config_contents
                else '-none-'
            ),
            ('data_file', self._data.filename if self._data else "-none-"),
            ('python', sys.version.replace('\n', '')),
            ('platform', platform.platform()),
            ('implementation', platform.python_implementation()),
            ('executable', sys.executable),
            ('def_encoding', sys.getdefaultencoding()),
            ('fs_encoding', sys.getfilesystemencoding()),
            ('pid', os.getpid()),
            ('cwd', os.getcwd()),
            ('path', sys.path),
            ('environment', sorted(
                ("%s = %s" % (k, v))
                for k, v in iitems(os.environ)
                if any(slug in k for slug in ("COV", "PY"))
            )),
            ('command_line', " ".join(getattr(sys, 'argv', ['-none-']))),
            ]

        if self._inorout:
            info.extend(self._inorout.sys_info())

        return info


# Mega debugging...
# $set_env.py: COVERAGE_DEBUG_CALLS - Lots and lots of output about calls to Coverage.
if int(os.environ.get("COVERAGE_DEBUG_CALLS", 0)):              # pragma: debugging
    from coverage.debug import decorate_methods, show_calls

    Coverage = decorate_methods(show_calls(show_args=True), butnot=['get_data'])(Coverage)


def process_startup():
    """Call this at Python start-up to perhaps measure coverage.

    If the environment variable COVERAGE_PROCESS_START is defined, coverage
    measurement is started.  The value of the variable is the config file
    to use.

    There are two ways to configure your Python installation to invoke this
    function when Python starts:

    #. Create or append to sitecustomize.py to add these lines::

        import coverage
        coverage.process_startup()

    #. Create a .pth file in your Python installation containing::

        import coverage; coverage.process_startup()

    Returns the :class:`Coverage` instance that was started, or None if it was
    not started by this call.

    """
    cps = os.environ.get("COVERAGE_PROCESS_START")
    if not cps:
        # No request for coverage, nothing to do.
        return None

    # This function can be called more than once in a process. This happens
    # because some virtualenv configurations make the same directory visible
    # twice in sys.path.  This means that the .pth file will be found twice,
    # and executed twice, executing this function twice.  We set a global
    # flag (an attribute on this function) to indicate that coverage.py has
    # already been started, so we can avoid doing it twice.
    #
    # https://bitbucket.org/ned/coveragepy/issue/340/keyerror-subpy has more
    # details.

    if hasattr(process_startup, "coverage"):
        # We've annotated this function before, so we must have already
        # started coverage.py in this process.  Nothing to do.
        return None

    cov = Coverage(config_file=cps)
    process_startup.coverage = cov
    cov._warn_no_data = False
    cov._warn_unimported_source = False
    cov._warn_preimported_source = False
    cov._auto_save = True
    cov.start()

    return cov


def _prevent_sub_process_measurement():
    """Stop any subprocess auto-measurement from writing data."""
    auto_created_coverage = getattr(process_startup, "coverage", None)
    if auto_created_coverage is not None:
        auto_created_coverage._auto_save = False
