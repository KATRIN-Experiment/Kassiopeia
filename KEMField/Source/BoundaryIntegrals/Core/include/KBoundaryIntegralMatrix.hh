#ifndef KBOUNDARYINTEGRALMATRIX_DEF
#define KBOUNDARYINTEGRALMATRIX_DEF

#include "KEMCoreMessage.hh"
#include "KMessageInterface.hh"
#include "KSquareMatrix.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KEMSimpleException.hh"


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
            if (fValueIsCached.size() <= i * fDimension + j) {
                throw KEMSimpleException("Internal error - Invalid cache size: No position " + std::to_string(i * fDimension + j));
            }

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

        if (sqrt(__UINT32_MAX__) < c.size()) {
            kem_cout(eError) << "Cannot create caches. Number of matrix elements exceeds numeric limits.\nMax size: "
                             << __UINT32_MAX__ << "\nAttempted size: " << (unsigned long long) c.size() * c.size()
                             << "\nPlease reduce the size of your problem or disable caching." << eom;
        }
        unsigned int num_elements = c.size() * c.size();
        unsigned int array_length = num_elements * basisDim2;
        unsigned int mem_size = array_length * (sizeof(ValueType) + sizeof(bool));

        if (num_elements > 16384)  // this would use about 2 GiB of RAM
        {
            kem_cout(eWarning) << "Resizing matrix cache to " << num_elements << " elements will use ca. " << mem_size/(1024*1024) << " MiB of memory" << eom;
        }

        fValueIsCached.resize(array_length, false);
        fCachedValue.resize(array_length);

        if (fValueIsCached.size() < array_length || fCachedValue.size() < array_length) {
            kem_cout(eError) << "Failed to resize matrix cache to " << num_elements << eom;
        }
    }
}

}  // namespace KEMField

#endif /* KVBOUNDARYINTEGRALMATRIX_DEF */
