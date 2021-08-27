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

template<> const KSLongLong KSLongLong::sZero(0);

template<> const KSFloat KSFloat::sZero(0.);

template<> const KSDouble KSDouble::sZero(0);

template<> const KSThreeVector KSThreeVector::sZero(KGeoBag::KThreeVector::sZero);

template<> const KSTwoVector KSTwoVector::sZero(KGeoBag::KTwoVector::sZero);

template<> const KSThreeMatrix KSThreeMatrix::sZero(KGeoBag::KThreeMatrix::sZero);

template<> const KSTwoMatrix KSTwoMatrix::sZero(KGeoBag::KTwoMatrix::sZero);

template<> const KSString KSString::sZero(std::string(""));

KSReadIterator::KSReadIterator() = default;

KSReadIterator::~KSReadIterator() = default;

}  // namespace Kassiopeia
