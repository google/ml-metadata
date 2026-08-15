# Third-party patches

This README documents the patch files available in this directory and should be used to document both existing patches and any future patches.

## Patch purposes

- `upb.patch`: Removes `-Werror=pedantic` from upb's Bazel C compiler options to avoid C23 extension build failures on modern Clang. Added to fix C23 extension errors during builds.
