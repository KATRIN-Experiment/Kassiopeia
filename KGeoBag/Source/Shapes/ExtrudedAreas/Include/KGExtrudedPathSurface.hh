#ifndef KGEXTRUDEDPATHSURFACE_HH_
#define KGEXTRUDEDPATHSURFACE_HH_

#include "KGArea.hh"
#include "KGPlanarPath.hh"
#include "KGShapeMessage.hh"

#include <memory>

namespace KGeoBag
{

template<class XPathType> class KGExtrudedPathSurface : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitExtrudedPathSurface(KGExtrudedPathSurface* aExtrudedPathSurface) = 0;
    };

  public:
    KGExtrudedPathSurface() :
        KGArea(),
        fPath(new XPathType()),
        fSign(1.),
        fZMin(0.),
        fZMax(0.),
        fExtrudedMeshCount(8),
        fExtrudedMeshPower(1.)
    {
        CompilerCheck();
    }
    KGExtrudedPathSurface(const KGExtrudedPathSurface& aCopy) :
        KGArea(aCopy),
        fPath(aCopy.fPath),
        fSign(aCopy.fSign),
        fZMin(aCopy.fZMin),
        fZMax(aCopy.fZMax),
        fExtrudedMeshCount(aCopy.fExtrudedMeshCount),
        fExtrudedMeshPower(aCopy.fExtrudedMeshPower)
    {}
    KGExtrudedPathSurface(const std::shared_ptr<XPathType>& aPath) :
        KGArea(),
        fPath(aPath),
        fSign(1.),
        fZMin(0.),
        fZMax(0.),
        fExtrudedMeshCount(8),
        fExtrudedMeshPower(1.)
    {}
    ~KGExtrudedPathSurface() override = default;

    static std::string Name()
    {
        return "extruded_" + XPathType::Name() + "_surface";
    }

  public:
    std::shared_ptr<XPathType>& Path()
    {
        return fPath;
    }
    const std::shared_ptr<XPathType>& Path() const
    {
        return fPath;
    }

    void Sign(const double& aSign)
    {
        fSign = aSign / fabs(aSign);
        return;
    }
    const double& Sign() const
    {
        return fSign;
    }

    void ZMin(const double& aZMin)
    {
        fZMin = aZMin;
        return;
    }
    const double& ZMin() const
    {
        return fZMin;
    }

    void ZMax(const double& aZMax)
    {
        fZMax = aZMax;
        return;
    }
    const double& ZMax() const
    {
        return fZMax;
    }

    void ExtrudedMeshCount(const unsigned int& aCount)
    {
        fExtrudedMeshCount = aCount;
        return;
    }
    const unsigned int& ExtrudedMeshCount() const
    {
        return fExtrudedMeshCount;
    }

    void ExtrudedMeshPower(const double& aPower)
    {
        fExtrudedMeshPower = aPower;
        return;
    }
    const double& ExtrudedMeshPower() const
    {
        return fExtrudedMeshPower;
    }

  protected:
    mutable std::shared_ptr<XPathType> fPath;
    mutable double fSign;
    mutable double fZMin;
    mutable double fZMax;
    mutable unsigned int fExtrudedMeshCount;
    mutable double fExtrudedMeshPower;

  public:
    void AreaInitialize() const override
    {
        return;
    }
    void AreaAccept(KGVisitor* aVisitor) override
    {
        shapemsg_debug("extruded path area named <" << GetName() << "> is receiving a visitor" << eom);
        auto* tExtrudedPathSurfaceVisitor = dynamic_cast<typename KGExtrudedPathSurface::Visitor*>(aVisitor);
        if (tExtrudedPathSurfaceVisitor != nullptr) {
            shapemsg_debug("extruded path area named <" << GetName() << "> is accepting a visitor" << eom);
            tExtrudedPathSurfaceVisitor->VisitExtrudedPathSurface(this);
        }
        return;
    }
    bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        double tXYAbove = fPath->Above(tXYPoint);
        if ((tXYAbove == true) && (fSign > 0.)) {
            return true;
        }
        else {
            return false;
        }
    }
    KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        KTwoVector tXYNearest = fPath->Point(tXYPoint);

        double tZ = aPoint.Z();
        if (tZ < fZMin) {
            tZ = fZMin;
        }
        if (tZ > fZMax) {
            tZ = fZMax;
        }

        return KGeoBag::KThreeVector(tXYNearest.X(), tXYNearest.Y(), tZ);
    }
    KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const override
    {
        KTwoVector tXYPoint = aPoint.ProjectXY();
        KTwoVector tXYNormal = fPath->Normal(tXYPoint);
        return KGeoBag::KThreeVector(fSign * tXYNormal.X(), fSign * tXYNormal.Y(), 0.);
    }

  private:
    static KGPlanarPath* CompilerCheck()
    {
        XPathType* tPath = nullptr;
        return tPath;
    }
};

}  // namespace KGeoBag

#endif
