# Updating vendored doctest

This repository vendors doctest as:

- `UnitTest/Doctest/include/doctest/doctest.h`
- `UnitTest/Doctest/src/doctest_main.cc`
- `UnitTest/Doctest/LICENSE.txt`

To update doctest:

1. Download the new upstream single-header release (`doctest.h`) from https://github.com/doctest/doctest/releases.
2. Replace `UnitTest/Doctest/include/doctest/doctest.h` with the new file.
3. Keep `UnitTest/Doctest/src/doctest_main.cc` as:
   - `#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`
   - `#include "doctest/doctest.h"`
4. Update `UnitTest/Doctest/LICENSE.txt` if upstream license text or copyright years changed.
5. Run the existing unit-test CI/build workflow to confirm compatibility.
