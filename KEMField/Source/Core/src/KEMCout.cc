#include "KEMCout.hh"

namespace KEMField
{
#ifdef KEMFIELD_SILENT
KDataDisplay<KNullStream> cout;
#else
#ifdef KEMFIELD_USE_KMESSAGE
KDataDisplay<KMessage_KEMField> cout;
#else
KDataDisplay<std::ostream> cout;
#endif
#endif
}  // namespace KEMField

namespace
{
bool EnableDebugOutput()
{
#ifdef KEMFIELD_USE_KMESSAGE
    // applications should set this themselves if that's what they need.
    // katrin::KMessageTable::GetInstance().SetTerminalVerbosity(katrin::eDebug);
#endif
    return true;
}

bool __attribute__((__unused__)) fEnableDebugOutput = EnableDebugOutput();
}  // namespace
