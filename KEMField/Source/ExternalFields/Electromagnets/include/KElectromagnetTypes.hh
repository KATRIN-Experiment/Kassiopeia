#ifndef KELECTROMAGNETTYPES_DEF
#define KELECTROMAGNETTYPES_DEF

#include "KTypelist.hh"

namespace KEMField
{
  class KLineCurrent;
  class KCoil;
  class KSolenoid;
  class KCurrentLoop;

  // A list of all of the electromagnet types
  typedef KTYPELIST_4( KLineCurrent,
		       KCoil,
		       KSolenoid,
		       KCurrentLoop) KElectromagnetTypes_;

  typedef NoDuplicates<KElectromagnetTypes_>::Result KElectromagnetTypes;
}

#include "KElectromagnetVisitor.hh"

#include "KLineCurrent.hh"
#include "KCurrentLoop.hh"
#include "KSolenoid.hh"
#include "KCoil.hh"


#endif /* KELECTROMAGNETTYPES_DEF */
