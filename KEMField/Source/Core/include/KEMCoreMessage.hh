#ifndef KEMCOREMESSAGE_HH_
#define KEMCOREMESSAGE_HH_

#include "KMessage.h"

KMESSAGE_DECLARE(KEMField, kem_cout)


#ifdef KEMField_ENABLE_DEBUG

#define kem_cout_debug(xCONTENT) kem_cout(katrin::eDebug) << xCONTENT;

#endif

#ifndef kem_cout_debug
#define kem_cout_debug(xCONTENT)
#endif

#endif
