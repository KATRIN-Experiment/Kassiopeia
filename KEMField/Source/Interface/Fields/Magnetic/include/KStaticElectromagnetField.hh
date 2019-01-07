/*
 * KStaticElectroMagnetField.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KSTATICELECTROMAGNETFIELD_HH_
#define KSTATICELECTROMAGNETFIELD_HH_

#include "KMagnetostaticField.hh"
#include "KMagneticFieldSolver.hh"
#include "KSmartPointer.hh"

namespace KEMField {

class KStaticElectromagnetField: public KMagnetostaticField {
public:
    KStaticElectromagnetField();
    virtual ~KStaticElectromagnetField();

    void SetDirectory( const std::string& aDirectory );
    void SetFile (const std::string& aFile );

    void SetFieldSolver( KSmartPointer<KMagneticFieldSolver> solver);
    KSmartPointer<KMagneticFieldSolver> GetFieldSolver();

    void SetContainer(KSmartPointer<KElectromagnetContainer> aContainer);
    KSmartPointer<KElectromagnetContainer> GetContainer() const;

protected:

    void InitializeCore();
    void CheckSolverExistance() const;

    KThreeVector MagneticPotentialCore(const KPosition& aSamplePoint) const;
    KThreeVector MagneticFieldCore(const KPosition& aSamplePoint) const;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const;
    std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const;

private:
    KSmartPointer<KElectromagnetContainer> fContainer;
    KSmartPointer<KMagneticFieldSolver> fFieldSolver;

    std::string fFile;
    std::string fDirectory;
};

} /* namespace KEMField */

#endif /* KSTATICELECTROMAGNETFIELD_HH_ */
