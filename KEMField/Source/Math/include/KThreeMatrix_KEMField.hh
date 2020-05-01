#ifndef KTHREEMATRIX_KEMFIELD_H_
#define KTHREEMATRIX_KEMFIELD_H_

#include "KThreeMatrix.hh"
#include "KThreeVector_KEMField.hh"

#include <cmath>

namespace KEMField
{
typedef KGeoBag::KThreeMatrix KThreeMatrix;

template<typename Stream> Stream& operator>>(Stream& s, KThreeMatrix& aThreeMatrix)
{
    s.PreStreamInAction(aThreeMatrix);
    s >> aThreeMatrix[0] >> aThreeMatrix[1] >> aThreeMatrix[2] >> aThreeMatrix[3] >> aThreeMatrix[4] >>
        aThreeMatrix[5] >> aThreeMatrix[6] >> aThreeMatrix[7] >> aThreeMatrix[8];
    s.PostStreamInAction(aThreeMatrix);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KThreeMatrix& aThreeMatrix)
{
    s.PreStreamOutAction(aThreeMatrix);
    s << aThreeMatrix[0] << aThreeMatrix[1] << aThreeMatrix[2] << aThreeMatrix[3] << aThreeMatrix[4] << aThreeMatrix[5]
      << aThreeMatrix[6] << aThreeMatrix[7] << aThreeMatrix[8];
    s.PostStreamOutAction(aThreeMatrix);
    return s;
}

/**
* @class KGradient
*
* @brief A class describing a field gradient.
*
* @author D.L. Furse
*/
class KGradient : public KThreeMatrix
{
  public:
    KGradient() : KThreeMatrix() {}
    KGradient(const KThreeMatrix& aMatrix) : KThreeMatrix(aMatrix) {}
    KGradient(const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY,
              const double& aYZ, const double& aZX, const double& aZY, const double& aZZ) :
        KThreeMatrix(anXX, anXY, anXZ, aYX, aYY, aYZ, aZX, aZY, aZZ)
    {}

    ~KGradient() override {}

    static std::string Name()
    {
        return "KGradient";
    }

    double& dFi_dxj(unsigned int i, unsigned int j)
    {
        return operator()(i, j);
    }
    const double& dFi_dxj(unsigned int i, unsigned int j) const
    {
        return operator()(i, j);
    }
};

template<typename Stream> Stream& operator>>(Stream& s, KGradient& aGradient)
{
    s.PreStreamInAction(aGradient);
    s >> aGradient[0] >> aGradient[1] >> aGradient[2] >> aGradient[3] >> aGradient[4] >> aGradient[5] >> aGradient[6] >>
        aGradient[7] >> aGradient[8];
    s.PostStreamInAction(aGradient);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KGradient& aGradient)
{
    s.PreStreamOutAction(aGradient);
    s << aGradient[0] << aGradient[1] << aGradient[2] << aGradient[3] << aGradient[4] << aGradient[5] << aGradient[6]
      << aGradient[7] << aGradient[8];
    s.PostStreamOutAction(aGradient);
    return s;
}

}  // namespace KEMField

#endif /* KTHREEMATRIX_KEMFIELD_H_ */
