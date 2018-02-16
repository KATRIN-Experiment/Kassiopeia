#ifndef KEMCOUT_DEF
#define KEMCOUT_DEF

#include "KDataDisplay.hh"

#ifdef KEMFIELD_USE_KMESSAGE
#include "KMessageInterface.hh"
#endif

namespace KEMField
{
#ifdef KEMFIELD_SILENT
  extern KDataDisplay<KNullStream> cout;
#else
#ifdef KEMFIELD_USE_KMESSAGE
  extern KDataDisplay<KMessage_KEMField> cout;
#else
  extern KDataDisplay<std::ostream> cout;
#endif
#endif
}

#endif /* KEMCOUT_DEF */
