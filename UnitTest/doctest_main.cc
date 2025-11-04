// Main file for doctest - provides main() function for all tests

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// global variables for argc/argv, used by some tests (e.g., MPI/PETSc initialization)
int __argc;
char** __argv;

// Custom main to set global variables
int main(int argc, char** argv) {
    __argc = argc;
    __argv = argv;
    
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    
    int res = context.run();
    
    return res;
}
