#ifndef KBOUNDARYINTEGRALMATRIX_DEF
#define KBOUNDARYINTEGRALMATRIX_DEF

#include "KEMCoreMessage.hh"
#include "KMessageInterface.hh"
#include "KSquareMatrix.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#define KEM_USE_CACHING true

namespace KEMField
{
template<class Integrator, bool enableCaching = false>
class KBoundaryIntegralMatrix : public KSquareMatrix<typename Integrator::Basis::ValueType>
{
  public:
    typedef typename Integrator::Basis::ValueType ValueType;

    KBoundaryIntegralMatrix(const KSurfaceContainer& c, Integrator& integrator);

    ~KBoundaryIntegralMatrix() override = default;

    unsigned int Dimension() const override
    {
        return fDimension;
    }

    const ValueType& operator()(unsigned int i, unsigned int j) const override
    {
        if (enableCaching) {
            if (fValueIsCached[i * fDimension + j])
                return fCachedValue[i * fDimension + j];
        }

        static ValueType value;

        value = fIntegrator.BoundaryIntegral(fContainer.at(j / Integrator::Basis::Dimension),
                                             j % Integrator::Basis::Dimension,
                                             fContainer.at(i / Integrator::Basis::Dimension),
                                             i % Integrator::Basis::Dimension);

        if (enableCaching) {
            fValueIsCached[i * fDimension + j] = true;
            fCachedValue[i * fDimension + j] = value;
        }

        return value;
    }

  protected:
    const KSurfaceContainer& fContainer;
    const unsigned int fDimension;
    Integrator& fIntegrator;

    mutable std::vector<bool> fValueIsCached;
    mutable std::vector<ValueType> fCachedValue;
};

template<class Integrator, bool enableCaching>
KBoundaryIntegralMatrix<Integrator, enableCaching>::KBoundaryIntegralMatrix(const KSurfaceContainer& c,
                                                                            Integrator& integrator) :
    KSquareMatrix<ValueType>(),
    fContainer(c),
    fDimension(c.size() * Integrator::Basis::Dimension),
    fIntegrator(integrator)
{
    if (enableCaching) {
        unsigned int basisDim2 = Integrator::Basis::Dimension;
        basisDim2 *= basisDim2;

        unsigned int num_elements = c.size() * c.size();

        if (num_elements > 16384)  // this would use about 2 GiB of RAM
        {
            kem_cout(eWarning) << "Resizing matrix cache to " << num_elements << " elements" << eom;
        }

        fValueIsCached.resize(num_elements * basisDim2, false);
        fCachedValue.resize(num_elements * basisDim2);
    }
}

}  // namespace KEMField

#endif /* KVBOUNDARYINTEGRALMATRIX_DEF */
