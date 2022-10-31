# Source files generated with cmake.
configure_out_srcs = [
    "build/libmariadb/ma_client_plugin.c",
]

# Header files generated with cmake.
configure_out_hdrs = [
    "build/include/ma_config.h",
    "build/include/config.h",
    "build/include/mariadb_version.h",
]

# A genrule to run cmake and generate configure_out_(srcs,hdrs).
genrule(
    name = "configure",
    srcs = glob(
        ["**"],
        exclude = ["bazel*/**"],
    ),
    outs = configure_out_srcs + configure_out_hdrs,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/build",
        "export TMP_DIR=$$(mktemp -d -t build.XXXXXX)",
        "mkdir -p $$TMP_DIR",
        "cp -R $$(pwd)/external/libmysqlclient/* $$TMP_DIR",
        "cd $$TMP_DIR",
        "mkdir build",
        "cd build",
        "cmake .. -DCMAKE_BUILD_TYPE=Release $${CMAKE_ICONV_FLAG-}",
        "cd ..",
        "cp -R ./build/* $$INSTALL_DIR",
        "rm -rf $$TMP_DIR",
    ]),
)

config_setting(
    name = "macos",
    constraint_values = [
        "@bazel_tools//platforms:osx",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libmysqlclient",
    srcs = configure_out_srcs + [
        # plugins.
        "plugins/auth/my_auth.c",
        "plugins/auth/old_password.c",
        "plugins/compress/c_zlib.c",
        "plugins/pvio/pvio_socket.c",
        # ssl.
        "libmariadb/secure/openssl.c",
        # core libmariadb
        "libmariadb/ma_array.c",
        "libmariadb/ma_charset.c",
        "libmariadb/ma_hashtbl.c",
        "libmariadb/ma_net.c",
        "libmariadb/mariadb_charset.c",
        "libmariadb/ma_time.c",
        "libmariadb/ma_default.c",
        "libmariadb/ma_errmsg.c",
        "libmariadb/mariadb_lib.c",
        "libmariadb/ma_list.c",
        "libmariadb/ma_pvio.c",
        "libmariadb/ma_tls.c",
        "libmariadb/ma_alloc.c",
        "libmariadb/ma_compress.c",
        "libmariadb/ma_init.c",
        "libmariadb/ma_password.c",
        "libmariadb/ma_ll2str.c",
        "libmariadb/ma_sha1.c",
        "libmariadb/mariadb_stmt.c",
        "libmariadb/ma_loaddata.c",
        "libmariadb/ma_stmt_codec.c",
        "libmariadb/ma_string.c",
        "libmariadb/ma_dtoa.c",
        "libmariadb/mariadb_dyncol.c",
        "libmariadb/mariadb_async.c",
        "libmariadb/ma_context.c",
        "libmariadb/ma_io.c",
    ],
    hdrs = configure_out_hdrs + glob([
        "include/*.h",
        "include/mysql/*.h",
        "include/mariadb/*.h",
    ]),
    copts = [
        "-g",
        "-DLIBMARIADB",
        "-DTHREAD",
        "-DHAVE_COMPRESS",
        "-DENABLED_LOCAL_INFILE",
        "-DLIBICONV_PLUG",
        "-DHAVE_OPENSSL",
        "-DHAVE_TLS",
    ] + select({
        ":macos": ["-D_XOPEN_SOURCE"],
        "//conditions:default": [],
    }),
    includes = [
        "build/include/",
        "include/",
        "include/mariadb/",
    ],
    linkopts = [
        "-lpthread",
        "-ldl",
        "-lm",
    ] + select({
        ":macos": ["-liconv"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@boringssl//:ssl",
        "@zlib",
    ],
)
