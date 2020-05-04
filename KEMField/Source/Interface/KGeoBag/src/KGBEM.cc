#include "KGBEM.hh"

namespace KGeoBag
{

template<> KGBEMData<KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary>::~KGBEMData() {}

template<> KGBEMData<KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary>::~KGBEMData() {}

template<> KGBEMData<KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary>::~KGBEMData() {}

template<> KGBEMData<KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary>::~KGBEMData() {}

}  // namespace KGeoBag
