#ifndef KZONALHARMONICCOEFFICIENTGENERATOR_H
#define KZONALHARMONICCOEFFICIENTGENERATOR_H

#include "KEMCoordinateSystem.hh"
#include "KEMCoreMessage.hh"
#include "KEMTicker.hh"
#include "KMessageInterface.hh"
#include "KZonalHarmonicSourcePoint.hh"
#include "KZonalHarmonicTypes.hh"

namespace KEMField
{
template<class Basis>
class KZHElementVisitorBase :
    public KSelectiveVisitor<typename KZonalHarmonicTrait<Basis>::Visitor, typename KZonalHarmonicTrait<Basis>::Types>
{
  public:
    using KSelectiveVisitor<typename KZonalHarmonicTrait<Basis>::Visitor,
                            typename KZonalHarmonicTrait<Basis>::Types>::Visit;

    ~KZHElementVisitorBase() override = default;

    virtual void SetGenerator(KZHCoefficientGeneratorElement* e) = 0;
};

template<typename Type, class Base> class KZHElementVisitorType : public Base
{
  public:
    using Base::Visit;
    ~KZHElementVisitorType() override = default;

    void Visit(Type& t) override
    {
        fGenerator.SetElement(&t);
        SetGenerator(&fGenerator);
    }

    void SetGenerator(KZHCoefficientGeneratorElement* e) override = 0;

  protected:
    KZHCoefficientGenerator<Type> fGenerator;
};

template<class Basis>
class KZonalHarmonicCoefficientGenerator :
    public KGenLinearHierarchy<typename KZonalHarmonicTrait<Basis>::Types, KZHElementVisitorType,
                               KZHElementVisitorBase<Basis>>
{
  public:
    using ZonalHarmonicType = KZonalHarmonicTrait<Basis>;
    using ElementContainer = typename ZonalHarmonicType::Container;

    using KGenLinearHierarchy<typename KZonalHarmonicTrait<Basis>::Types, KZHElementVisitorType,
                              KZHElementVisitorBase<Basis>>::Visit;

    KZonalHarmonicCoefficientGenerator(ElementContainer& container) : fElementContainer(container) {}
    ~KZonalHarmonicCoefficientGenerator() override = default;

    void GroupCoaxialElements(std::vector<ElementContainer*>& subcontainers, double coaxialityTolerance);
    void BifurcateElements(std::vector<ElementContainer*>& subcontainers);

    void GenerateCentralSourcePointsByFixedDistance(std::vector<KZonalHarmonicSourcePoint*>& sps, unsigned int nCoeffs,
                                                    double deltaZ, double z1 = 0., double z2 = 0.);
    void GenerateCentralSourcePointsByFractionalDistance(std::vector<KZonalHarmonicSourcePoint*>& sps,
                                                         unsigned int nCoeffs, double fractionalDistance = 0.2,
                                                         double deltaZ = 0., double z1 = 0., double z2 = 0.);
    void GenerateRemoteSourcePoints(std::vector<KZonalHarmonicSourcePoint*>& sps, unsigned int nCoeffs,
                                    unsigned int nSPs, double z1 = 0., double z2 = 0.);
    const KEMCoordinateSystem* GetCoordinateSystem();

  protected:
    void SetGenerator(KZHCoefficientGeneratorElement* e) override
    {
        fGenerator = e;
    }

    KZonalHarmonicSourcePoint* GenerateCentralSourcePoint(double z, unsigned int nCoeffs);

    KZonalHarmonicSourcePoint* GenerateRemoteSourcePoint(double z, unsigned int nCoeffs);

    double ComputeCentralRho(double z);
    double ComputeRemoteRho(double z);

    void SourcePointExtrema(double& z1, double& z2);

    void ComputeCentralCoefficients(double z0, double rho, std::vector<double>& coeffs);
    void ComputeRemoteCoefficients(double z0, double rho, std::vector<double>& coeffs);

    ElementContainer& fElementContainer;
    KZHCoefficientGeneratorElement* fGenerator;
};

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::GroupCoaxialElements(
    std::vector<typename ZonalHarmonicType::Container*>& subcontainers, double coaxialityTolerance)
{
    using ElementContainer = typename ZonalHarmonicType::Container;

    if (fElementContainer.empty())
        return;

    auto* nonAxiallySymmetricElements = new ElementContainer();

    std::vector<KEMCoordinateSystem> coordinateSystems;

    subcontainers.push_back(new ElementContainer());
    subcontainers.back()->IsOwner(false);

    unsigned int element = 0;
    unsigned int grouped_elements = 0;

    for (; element < fElementContainer.size(); element++) {
        fGenerator = nullptr;
        fElementContainer.at(element)->Accept(*this);

        if (fGenerator) {
            subcontainers.back()->push_back(fElementContainer.at(element));
            coordinateSystems.push_back(KEMCoordinateSystem(fGenerator->GetCoordinateSystem()));
            break;
        }
        else
            nonAxiallySymmetricElements->push_back(fElementContainer.at(element));
    }

    for (unsigned int i = element + 1; i < fElementContainer.size(); i++) {
        bool newCoordinateSystem = true;
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);

        if (fGenerator) {
            for (unsigned int j = 0; j < coordinateSystems.size(); j++) {
                if (fGenerator->IsCoaxial(coordinateSystems.at(j), coaxialityTolerance)) {
                    subcontainers.at(j)->push_back(fElementContainer.at(i));
                    newCoordinateSystem = false;
                    grouped_elements++;
                }
            }
        }
        else {
            newCoordinateSystem = false;
            nonAxiallySymmetricElements->push_back(fElementContainer.at(i));
        }

        if (newCoordinateSystem) {
            subcontainers.push_back(new ElementContainer());
            subcontainers.back()->IsOwner(false);
            subcontainers.back()->push_back(fElementContainer.at(i));
            coordinateSystems.push_back(KEMCoordinateSystem(fGenerator->GetCoordinateSystem()));
        }
    }

    if (grouped_elements > 0) {
        kem_cout(eWarning) << "Added " << grouped_elements << " elements to existing coaxial group (coax. tolerance "
                           << coaxialityTolerance << ")" << eom;
    }

    if (!(nonAxiallySymmetricElements->empty())) {
        subcontainers.push_back(nonAxiallySymmetricElements);
        subcontainers.back()->IsOwner(false);
    }
    else
        delete nonAxiallySymmetricElements;
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::BifurcateElements(
    std::vector<typename ZonalHarmonicType::Container*>& subcontainers)
{
    using ElementContainer = typename ZonalHarmonicType::Container;

    if (fElementContainer.size() < 2)
        return;

    double zMid;
    double z1 = 0;
    double z2 = 0;
    SourcePointExtrema(z1, z2);
    zMid = (z1 + z2) * .5;

    subcontainers.push_back(new ElementContainer());
    subcontainers.back()->IsOwner(false);
    subcontainers.push_back(new ElementContainer());
    subcontainers.back()->IsOwner(false);

    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();

    for (unsigned int i = 0; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double offset = fGenerator->AxialOffset(*coordinateSystem);
            fGenerator->GetExtrema(z1, z2);
            double z = (z1 + z2) * .5;

            if (z + offset < zMid)
                subcontainers.front()->push_back(fElementContainer.at(i));
            else
                subcontainers.back()->push_back(fElementContainer.at(i));
        }
    }

    if (subcontainers.front()->empty()) {
        delete subcontainers.front();
        subcontainers.erase(subcontainers.begin());
    }

    if (subcontainers.back()->empty()) {
        delete subcontainers.back();
        subcontainers.erase(--subcontainers.end());
    }
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::GenerateCentralSourcePointsByFixedDistance(
    std::vector<KZonalHarmonicSourcePoint*>& sps, unsigned int nCoeffs, double deltaZ, double z1, double z2)
{
    sps.clear();

    // check if the container holds non-axially symmetric elements
    {
        fGenerator = nullptr;
        fElementContainer.at(0)->Accept(*this);

        if (!fGenerator)
            return;
    }

    if (z2 < z1) {
        double tmp = z1;
        z1 = z2;
        z2 = tmp;
    }

    if (fabs(z1 - z2) < 1.e-10)
        SourcePointExtrema(z1, z2);

    unsigned int nSPs = (z2 - z1) / deltaZ;
    if (nSPs == 0)
        nSPs = 1;

    kem_cout(eNormal) << "Computing " << nSPs << " central source points for " << Basis::Name()
                      << " along the local z-axis from " << z1 << " to " << z2 << "." << eom;

    KTicker ticker;
    ticker.StartTicker(nSPs);

    for (unsigned int i = 0; i < nSPs; i++) {
        ticker.Tick(i);
        double z = z1 + deltaZ * i;
        sps.push_back(GenerateCentralSourcePoint(z, nCoeffs));
    }

    ticker.EndTicker();
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::GenerateCentralSourcePointsByFractionalDistance(
    std::vector<KZonalHarmonicSourcePoint*>& sps, unsigned int nCoeffs, double fractionalDistance, double deltaZ,
    double z1, double z2)
{
    sps.clear();

    // check if the container holds non-axially symmetric elements
    {
        fGenerator = nullptr;
        fElementContainer.at(0)->Accept(*this);

        if (!fGenerator)
            return;
    }

    if (z2 < z1) {
        double tmp = z1;
        z1 = z2;
        z2 = tmp;
    }

    if (fabs(z1 - z2) < 1.e-10)
        SourcePointExtrema(z1, z2);

    kem_cout(eNormal) << "Computing central source points for " << Basis::Name() << " along the local z-axis from "
                      << z1 << " to " << z2 << "." << eom;

    KTicker ticker;
    ticker.StartTicker(z2 - z1);

    sps.push_back(GenerateCentralSourcePoint(z1, nCoeffs));

    int counter = 0;
    double z = z1;
    while (true) {
        ticker.Tick(z - z1);
        double dZ = fractionalDistance * ComputeCentralRho(z);
        if (dZ < deltaZ)
            dZ = deltaZ;
        z += dZ;
        if (z >= z2)
            break;
        counter++;
        sps.push_back(GenerateCentralSourcePoint(z, nCoeffs));
    }

    sps.push_back(GenerateCentralSourcePoint(z2, nCoeffs));

    kem_cout(eNormal) << counter << " central source points have been computed" << eom;

    ticker.EndTicker();
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::GenerateRemoteSourcePoints(std::vector<KZonalHarmonicSourcePoint*>& sps,
                                                                           unsigned int nCoeffs, unsigned int nSPs,
                                                                           double z1, double z2)
{
    sps.clear();

    // check if the container holds non-axially symmetric elements
    {
        fGenerator = nullptr;
        fElementContainer.at(0)->Accept(*this);

        if (!fGenerator)
            return;
    }

    if (z2 < z1) {
        double tmp = z1;
        z1 = z2;
        z2 = tmp;
    }

    if (fabs(z1 - z2) < 1.e-10)
        SourcePointExtrema(z1, z2);

    double deltaZ = (z2 - z1) / (nSPs - 1);

    // if the z positions are still the same, the geometry consists of a single
    // ring and needs only one source point.
    if (fabs(z1 - z2) < 1.e-10) {
        nSPs = 1;
        deltaZ = 0;
    }

    kem_cout(eNormal) << "Computing " << nSPs << " remote source points for " << Basis::Name()
                      << " along the local z-axis from " << z1 << " to " << z2 << "." << eom;

    KTicker ticker;
    ticker.StartTicker(nSPs);

    for (unsigned int i = 0; i < nSPs; i++) {
        ticker.Tick(i);
        double z = z1 + deltaZ * i;
        sps.push_back(GenerateRemoteSourcePoint(z, nCoeffs));
    }

    ticker.EndTicker();
}

template<class Basis> const KEMCoordinateSystem* KZonalHarmonicCoefficientGenerator<Basis>::GetCoordinateSystem()
{
    if (!(fElementContainer.empty())) {
        //   fElementContainer.at(0)->Accept(*this);
        unsigned int elem = 0;
        fGenerator = nullptr;
        do {
            fElementContainer.at(elem++)->Accept(*this);
        } while (fGenerator == nullptr && elem < fElementContainer.size());

        if (fGenerator) {
            fGenerator->GetCoordinateSystem();
            return &(fGenerator->GetCoordinateSystem());
        }
    }
    return &gGlobalCoordinateSystem;
}

template<class Basis>
KZonalHarmonicSourcePoint* KZonalHarmonicCoefficientGenerator<Basis>::GenerateCentralSourcePoint(double z,
                                                                                                 unsigned int nCoeffs)
{
    double rho = ComputeCentralRho(z);
    std::vector<double> coeffs(nCoeffs, 0.);

    ComputeCentralCoefficients(z, rho, coeffs);

    auto* sp = new KZonalHarmonicSourcePoint();
    sp->SetValues(z, rho, coeffs);

    return sp;
}

template<class Basis>
KZonalHarmonicSourcePoint* KZonalHarmonicCoefficientGenerator<Basis>::GenerateRemoteSourcePoint(double z,
                                                                                                unsigned int nCoeffs)
{
    double rho = ComputeRemoteRho(z);
    std::vector<double> coeffs(nCoeffs, 0.);

    ComputeRemoteCoefficients(z, rho, coeffs);

    auto* sp = new KZonalHarmonicSourcePoint();
    sp->SetValues(z, rho, coeffs);

    return sp;
}

template<class Basis> double KZonalHarmonicCoefficientGenerator<Basis>::ComputeCentralRho(double z)
{
    double rho = 1.e10;
    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();
    for (unsigned int i = 0; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double offset = fGenerator->AxialOffset(*coordinateSystem);
            double tmp = fGenerator->ComputeRho(z - offset, true);
            if (tmp < rho)
                rho = tmp;
        }
    }
    return rho;
}

template<class Basis> double KZonalHarmonicCoefficientGenerator<Basis>::ComputeRemoteRho(double z)
{
    double rho = 0.;
    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();
    for (unsigned int i = 0; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double offset = fGenerator->AxialOffset(*coordinateSystem);
            double tmp = fGenerator->ComputeRho(z - offset, false);
            if (tmp > rho)
                rho = tmp;
        }
    }
    return rho;
}

template<class Basis> void KZonalHarmonicCoefficientGenerator<Basis>::SourcePointExtrema(double& z1, double& z2)
{
    if (fElementContainer.empty())
        return;

    unsigned int elem = 0;
    fGenerator = nullptr;
    do {
        fElementContainer.at(elem++)->Accept(*this);
    } while (fGenerator == nullptr && elem < fElementContainer.size());

    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();

    if (fGenerator == nullptr) {
        return;
    }

    fGenerator->GetExtrema(z1, z2);

    for (unsigned int i = 1; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double tmp1, tmp2;
            fGenerator->GetExtrema(tmp1, tmp2);
            double offset = fGenerator->AxialOffset(*coordinateSystem);

            if (tmp1 + offset < z1)
                z1 = tmp1 + offset;
            if (tmp2 + offset > z2)
                z2 = tmp2 + offset;
        }
    }
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::ComputeCentralCoefficients(double z0, double rho,
                                                                           std::vector<double>& coeffs)
{
    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();

    for (unsigned int i = 0; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double offset = fGenerator->AxialOffset(*coordinateSystem);
            fGenerator->ComputeCentralCoefficients(z0 - offset, rho, coeffs);
        }
    }
}

template<class Basis>
void KZonalHarmonicCoefficientGenerator<Basis>::ComputeRemoteCoefficients(double z0, double rho,
                                                                          std::vector<double>& coeffs)
{
    const KEMCoordinateSystem* coordinateSystem = GetCoordinateSystem();

    for (unsigned int i = 0; i < fElementContainer.size(); i++) {
        fGenerator = nullptr;
        fElementContainer.at(i)->Accept(*this);
        if (fGenerator) {
            double offset = fGenerator->AxialOffset(*coordinateSystem);
            fGenerator->ComputeRemoteCoefficients(z0 - offset, rho, coeffs);
        }
    }
}
}  // namespace KEMField

#endif /* KZONALHARMONICCOEFFICIENTGENERATOR */
