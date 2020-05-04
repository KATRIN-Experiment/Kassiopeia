#include "KSReadIterator.h"

namespace Kassiopeia
{

template<> const KSBool KSBool::sZero(false);

template<> const KSUChar KSUChar::sZero(0);

template<> const KSChar KSChar::sZero(0);

template<> const KSUShort KSUShort::sZero(0);

template<> const KSShort KSShort::sZero(0);

template<> const KSUInt KSUInt::sZero(0);

template<> const KSInt KSInt::sZero(0);

template<> const KSULong KSULong::sZero(0);

template<> const KSLong KSLong::sZero(0);

template<> const KSFloat KSFloat::sZero(0.);

template<> const KSDouble KSDouble::sZero(0);

template<> const KSThreeVector KSThreeVector::sZero(KThreeVector(0., 0., 0.));

template<> const KSTwoVector KSTwoVector::sZero(KTwoVector(0., 0.));

template<> const KSString KSString::sZero(string(""));

KSReadIterator::KSReadIterator() {}

KSReadIterator::~KSReadIterator() {}

}  // namespace Kassiopeia
