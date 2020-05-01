#ifndef KZONALHARMONICCONTAINER_DEF
#define KZONALHARMONICCONTAINER_DEF

#include "KEMCoordinateSystem.hh"
#include "KMD5HashGenerator.hh"
#include "KZonalHarmonicCoefficientGenerator.hh"
#include "KZonalHarmonicParameters.hh"
#include "KZonalHarmonicSourcePoint.hh"

namespace KEMField
{
template<class Basis> class KZonalHarmonicContainer
{
  public:
    typedef KZonalHarmonicTrait<Basis> ZonalHarmonicType;
    typedef typename ZonalHarmonicType::Container ElementContainer;
    typedef std::vector<KZonalHarmonicSourcePoint*> SourcePointVector;
    typedef std::vector<KZonalHarmonicContainer<Basis>*> ZonalHarmonicContainerVector;

    KZonalHarmonicContainer(ElementContainer& elementContainer, KZonalHarmonicParameters* parameters = nullptr);

    virtual ~KZonalHarmonicContainer();

    static std::string Name()
    {
        return std::string("ZonalHarmonicContainer_") + ZonalHarmonicType::Name();
    }

    void ComputeCoefficients()
    {
        ComputeCoefficients(-1);
    }

    ElementContainer& GetElementContainer()
    {
        return fElementContainer;
    }
    const ElementContainer& GetElementContainer() const
    {
        return fElementContainer;
    }
    const KEMCoordinateSystem& GetCoordinateSystem() const
    {
        return fCoordinateSystem;
    }
    SourcePointVector& GetCentralSourcePoints()
    {
        return fCentralSourcePoints;
    }
    const SourcePointVector& GetCentralSourcePoints() const
    {
        return fCentralSourcePoints;
    }
    SourcePointVector& GetRemoteSourcePoints()
    {
        return fRemoteSourcePoints;
    }
    const SourcePointVector& GetRemoteSourcePoints() const
    {
        return fRemoteSourcePoints;
    }
    ZonalHarmonicContainerVector& GetSubContainers()
    {
        return fSubContainers;
    }
    const ZonalHarmonicContainerVector& GetSubContainers() const
    {
        return fSubContainers;
    }
    KZonalHarmonicParameters& GetParameters()
    {
        return *fParameters;
    }
    const KZonalHarmonicParameters& GetParameters() const
    {
        return *fParameters;
    }

  private:
    KZonalHarmonicContainer() {}

    void ComputeCoefficients(int level);
    void ConstructSubContainers(int level = -1);

    ElementContainer& fElementContainer;
    KEMCoordinateSystem fCoordinateSystem;
    SourcePointVector fCentralSourcePoints;
    SourcePointVector fRemoteSourcePoints;
    ZonalHarmonicContainerVector fSubContainers;
    KZonalHarmonicParameters* fParameters;

    bool fHead;

    template<typename Stream> friend Stream& operator>>(Stream& s, KZonalHarmonicContainer<Basis>& c)
    {
        s.PreStreamInAction(c);

        // Compare the hashes of the element container
        std::string hash;
        s >> hash;
        KMD5HashGenerator hashGenerator;
        if (hash != hashGenerator.GenerateHash(c.fElementContainer)) {
            std::cout << "Error!  Hashes don't match." << std::endl;
            return s;
        }

        // Get the coordinate system
        s >> c.fCoordinateSystem;

        // Get the parameters
        if (c.fHead) {
            s >> *(c.fParameters);

            // Resize the coefficient vectors
            KZHLegendreCoefficients::GetInstance()->InitializeLegendrePolynomialArrays(
                (c.fParameters->GetNCentralCoefficients() > c.fParameters->GetNRemoteCoefficients()
                     ? c.fParameters->GetNCentralCoefficients()
                     : c.fParameters->GetNRemoteCoefficients()));

            // recreate the subcontainers with new parameters
            for (auto it = c.fSubContainers.begin(); it != c.fSubContainers.end(); ++it)
                delete *it;
            c.fSubContainers.clear();
            c.ConstructSubContainers();
        }

        // Get the central source points
        unsigned int nElements;
        s >> nElements;
        for (unsigned int i = 0; i < c.fCentralSourcePoints.size(); i++)
            delete c.fCentralSourcePoints.at(i);
        c.fCentralSourcePoints.clear();
        for (unsigned int i = 0; i < nElements; i++) {
            auto* sP = new KZonalHarmonicSourcePoint();
            s >> *sP;
            c.fCentralSourcePoints.push_back(sP);
        }

        // Get the remote source points
        s >> nElements;
        for (unsigned int i = 0; i < c.fRemoteSourcePoints.size(); i++)
            delete c.fRemoteSourcePoints.at(i);
        c.fRemoteSourcePoints.clear();
        for (unsigned int i = 0; i < nElements; i++) {
            auto* sP = new KZonalHarmonicSourcePoint();
            s >> *sP;
            c.fRemoteSourcePoints.push_back(sP);
        }

        // Get the subcontainers
        s >> nElements;
        if (nElements != c.fSubContainers.size()) {
            std::cout << "Error!  Subcontainer sizes don't match." << std::endl;
            return s;
        }
        for (unsigned int i = 0; i < nElements; i++)
            s >> *(c.fSubContainers.at(i));

        s.PostStreamInAction(c);
        return s;
    }

    template<typename Stream> friend Stream& operator<<(Stream& s, const KZonalHarmonicContainer<Basis>& c)
    {
        s.PreStreamOutAction(c);

        // Send the hash
        KMD5HashGenerator hashGenerator;
        s << hashGenerator.GenerateHash(c.fElementContainer);

        // Send the coordinate system
        s << c.fCoordinateSystem;

        // Send the parameters
        if (c.fHead)
            s << *(c.fParameters);

        // Send the central source points
        s << (unsigned int) (c.fCentralSourcePoints.size());
        for (unsigned int i = 0; i < c.fCentralSourcePoints.size(); i++)
            s << *(c.fCentralSourcePoints.at(i));

        // Send the remote source points
        s << (unsigned int) (c.fRemoteSourcePoints.size());
        for (unsigned int i = 0; i < c.fRemoteSourcePoints.size(); i++)
            s << *(c.fRemoteSourcePoints.at(i));

        // Send the subcontainers
        s << (unsigned int) (c.fSubContainers.size());
        for (unsigned int i = 0; i < c.fSubContainers.size(); i++)
            s << *(c.fSubContainers.at(i));

        s.PostStreamOutAction(c);
        return s;
    }
};

template<class Basis>
KZonalHarmonicContainer<Basis>::KZonalHarmonicContainer(ElementContainer& elementContainer,
                                                        KZonalHarmonicParameters* parameters) :
    fElementContainer(elementContainer),
    fCoordinateSystem(gGlobalCoordinateSystem),
    fParameters(parameters),
    fHead(true)
{
    if (!fParameters)
        fParameters = new KZonalHarmonicParameters();
}

template<class Basis> KZonalHarmonicContainer<Basis>::~KZonalHarmonicContainer()
{
    for (auto it = fCentralSourcePoints.begin(); it != fCentralSourcePoints.end(); ++it)
        delete *it;

    for (auto it = fRemoteSourcePoints.begin(); it != fRemoteSourcePoints.end(); ++it)
        delete *it;

    if (fHead) {
        delete fParameters;
        fParameters = nullptr;
    }
    else
        fElementContainer.~ElementContainer();

    for (auto it = fSubContainers.begin(); it != fSubContainers.end(); ++it)
        delete *it;
}

template<class Basis> void KZonalHarmonicContainer<Basis>::ConstructSubContainers(int level)
{
    if (fElementContainer.empty())
        return;

    KZonalHarmonicCoefficientGenerator<Basis> coefficientGenerator(fElementContainer);

    std::vector<typename ZonalHarmonicType::Container*> subcontainers;

    if (level == -1) {
        // split according to axial symmetry
        coefficientGenerator.GroupCoaxialElements(subcontainers, fParameters->GetCoaxialityTolerance());

        if (subcontainers.size() > 1) {
            for (unsigned int i = 0; i < subcontainers.size(); i++) {
                fSubContainers.push_back(new KZonalHarmonicContainer<Basis>(*(subcontainers.at(i))));
                fSubContainers.back()->fHead = false;
                fSubContainers.back()->fParameters = fParameters;
                fSubContainers.back()->ConstructSubContainers(level + 1);
            }
            return;
        }
    }

    // split according to bifurcation
    if (level < (int) (fParameters->GetNBifurcations())) {
        std::vector<typename ZonalHarmonicType::Container*> subcontainers;
        coefficientGenerator.BifurcateElements(subcontainers);

        if (subcontainers.size() > 1) {
            for (unsigned int i = 0; i < subcontainers.size(); i++) {
                fSubContainers.push_back(new KZonalHarmonicContainer<Basis>(*(subcontainers.at(i))));
                fSubContainers.back()->fHead = false;
                fSubContainers.back()->fParameters = fParameters;
                fSubContainers.back()->ConstructSubContainers(level + 1);
            }
        }
    }
}

template<class Basis> void KZonalHarmonicContainer<Basis>::ComputeCoefficients(int level)
{
    if (fElementContainer.empty())
        return;

    KZHLegendreCoefficients::GetInstance()->InitializeLegendrePolynomialArrays(
        (fParameters->GetNCentralCoefficients() > fParameters->GetNRemoteCoefficients()
             ? fParameters->GetNCentralCoefficients()
             : fParameters->GetNRemoteCoefficients()));

    if (level == -1) {
        for (auto it = fSubContainers.begin(); it != fSubContainers.end(); ++it)
            delete *it;
        fSubContainers.clear();
        ConstructSubContainers();
    }

    if ((level == -1 && fSubContainers.size() == 0) || level != -1) {
        KZonalHarmonicCoefficientGenerator<Basis> coefficientGenerator(fElementContainer);

        // user-defined source-point extrema are only valid in the top-level coordinate system
        double z1 = 0.;
        double z2 = 0.;
        if (level == -1) {
            z1 = fParameters->GetCentralZ1();
            z2 = fParameters->GetCentralZ2();
        }

        if (fParameters->GetCentralFractionalSpacing()) {
            coefficientGenerator.GenerateCentralSourcePointsByFractionalDistance(
                fCentralSourcePoints,
                fParameters->GetNCentralCoefficients(),
                fParameters->GetCentralFractionalDistance(),
                fParameters->GetCentralDeltaZ(),
                z1,
                z2);
        }
        else {
            coefficientGenerator.GenerateCentralSourcePointsByFixedDistance(fCentralSourcePoints,
                                                                            fParameters->GetNCentralCoefficients(),
                                                                            fParameters->GetCentralDeltaZ(),
                                                                            z1,
                                                                            z2);
        }

        z1 = z2 = 0.;
        if (level == -1) {
            z1 = fParameters->GetRemoteZ1();
            z2 = fParameters->GetRemoteZ2();
        }

        coefficientGenerator.GenerateRemoteSourcePoints(fRemoteSourcePoints,
                                                        fParameters->GetNRemoteCoefficients(),
                                                        fParameters->GetNRemoteSourcePoints(),
                                                        z1,
                                                        z2);

        fCoordinateSystem = *(coefficientGenerator.GetCoordinateSystem());
    }

    if (!fSubContainers.empty()) {
        KEMField::cout << "Computing source points for " << fSubContainers.size() << " subcontainers at level "
                       << level + 1 << KEMField::endl;

        for (auto it = fSubContainers.begin(); it != fSubContainers.end(); ++it)
            (*it)->ComputeCoefficients(level + 1);
    }
}

}  // end namespace KEMField

#endif /* KZONALHARMONICCONTAINER_DEF */
