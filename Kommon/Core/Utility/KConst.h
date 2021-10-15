/**
 * @file KConst.h
 * @author J. Behrens
 */

// This is now a wrapper file for the actual constants definitions.

#if KConst_REFERENCE_EPOCH == 2006
#include "KConst_2006.h"
#elif KConst_REFERENCE_EPOCH == 2021
#include "KConst_2021.h"
#else
#error "Unsupported value for KConst_REFERENCE_EPOCH."
#endif
