# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sets up a local repository to statically link system provided mysqlclient libs.

Clients can then depend on `@libmysqlclient` to use the MYSQL C API.
"""

def mysql_configure_fail(msg):
    """Output failure message when mysql configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sMYSQL-Configuration Error:%s %s\n" % (red, no_color, msg))

def _run_command(ctx, args, fail_on_exec_error = True):
    """Runs the command represented by `args`.

    Args:
      ctx: A `repository_ctx` with which the command should be executed.
      args: A list of `str`. First element represents the program to run.
            (Must not be empty).
      fail_on_exec_error: A boolean. If true, any error from the command
                          execution causes the extension to fail.

    Returns:
      A tuple of (bool, str). The bool indicates whether the command
      returned success or failure. The str is the stdout of the command
      on success or an empty string otherwise.
    """
    if len(args) == 0:
        mysql_configure_fail("Internal error: args must not be empty")

    cmd = ctx.which(args[0])
    if cmd == None:
        mysql_configure_fail("%s not found" % args[0])

    result = ctx.execute([cmd.realpath] + args[1:])
    if result.return_code and fail_on_exec_error:
        mysql_configure_fail("%s failed, stderr %s" % (args, result.stderr))

    if result.return_code:
        return False, ""
    else:
        return True, result.stdout.strip()

def _mysql_client_lib(ctx):
    """Sets up a rule to statically link system provided mysqlclient libs."""
    _, pkglibdir = _run_command(ctx, ["mysql_config", "--variable=pkglibdir"])

    libpath = None
    for lib_name in ["libmysqlclient.a", "libmariadbclient.a"]:
        path = "%s/%s" % (pkglibdir, lib_name)
        ok, _ = _run_command(ctx, ["ls", path], fail_on_exec_error = False)
        if ok:
            libpath = path
            break

    if libpath == None:
        mysql_configure_fail("no lib(mysql|mariadb)client.a in %s" % pkglibdir)

    ctx.symlink(libpath, "libmysqlclient.a")
    build_lines = [
        "cc_library(",
        "  name = \"libmysqlclient\",",
        "  srcs = [\"libmysqlclient.a\"],",
        "  visibility = [\"//visibility:public\"],",
        ")",
    ]
    ctx.file("BUILD", content = "\n".join(build_lines), executable = False)

mysql_client_lib = repository_rule(
    implementation = _mysql_client_lib,
    local = True,
)

def mysql_configure():
    mysql_client_lib(name = "libmysqlclient")
