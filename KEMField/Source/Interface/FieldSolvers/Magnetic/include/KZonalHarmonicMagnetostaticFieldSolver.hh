/*
 * KZonalHarmonicMagnetostaticFieldSolver.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_
#define KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_

#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KMagneticFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"

namespace KEMField
{

class KZonalHarmonicMagnetostaticFieldSolver : public KMagneticFieldSolver
{
  public:
    typedef std::vector<KZonalHarmonicSourcePoint*> SourcePointVector;

    KZonalHarmonicMagnetostaticFieldSolver();
    ~KZonalHarmonicMagnetostaticFieldSolver() override;

    bool UseCentralExpansion(const KPosition& P);
    bool UseRemoteExpansion(const KPosition& P);

    std::set<std::pair<double, double>> CentralSourcePoints();
    std::set<std::pair<double, double>> RemoteSourcePoints();

    void InitializeCore(KElectromagnetContainer& container) override;

    KFieldVector MagneticPotentialCore(const KPosition& P) const override;
    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;
    std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const override;

    KZonalHarmonicParameters* GetParameters()
    {
        return fParameters.get();
    }


  private:
    KElectromagnetIntegrator fIntegrator;
    KZonalHarmonicContainer<KMagnetostaticBasis>* fZHContainer;
    KZonalHarmonicFieldSolver<KMagnetostaticBasis>* fZonalHarmonicFieldSolver;
    std::shared_ptr<KZonalHarmonicParameters> fParameters;
};

} /* namespace KEMField */

#endif /* KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_ */
