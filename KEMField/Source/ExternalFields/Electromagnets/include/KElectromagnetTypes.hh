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
using KElectromagnetTypes_ = KEMField::KTypelist<
    KLineCurrent,
    KEMField::KTypelist<KCoil, KEMField::KTypelist<KSolenoid, KEMField::KTypelist<KCurrentLoop, KEMField::KNullType>>>>;

using KElectromagnetTypes = NoDuplicates<KElectromagnetTypes_>::Result;
}  // namespace KEMField

#include "KCoil.hh"
#include "KCurrentLoop.hh"
#include "KElectromagnetVisitor.hh"
#include "KLineCurrent.hh"
#include "KSolenoid.hh"


#endif /* KELECTROMAGNETTYPES_DEF */
