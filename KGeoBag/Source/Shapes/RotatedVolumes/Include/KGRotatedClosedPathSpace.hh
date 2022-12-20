#ifndef KGROTATEDCLOSEDPATHSPACE_HH_
#define KGROTATEDCLOSEDPATHSPACE_HH_

#include "KGPlanarClosedPath.hh"
#include "KGRotatedPathSurface.hh"
#include "KGShapeMessage.hh"
#include "KGVolume.hh"

namespace KGeoBag
{

template<class XPathType> class KGRotatedClosedPathSpace : public KGVolume
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitRotatedClosedPathSpace(KGRotatedClosedPathSpace* aRotatedClosedPathSpace) = 0;
    };

  public:
    KGRotatedClosedPathSpace() : KGVolume(), fPath(new XPathType()), fRotatedMeshCount(64)
    {
        CompilerCheck();
    }
    KGRotatedClosedPathSpace(const KGRotatedClosedPathSpace<XPathType>& aCopy) :
        KGVolume(aCopy),
        fPath(aCopy.fPath),
        fRotatedMeshCount(aCopy.fRotatedMeshCount)
    {}
    KGRotatedClosedPathSpace(const std::shared_ptr<XPathType>& aPath) : KGVolume(), fPath(aPath), fRotatedMeshCount(64)
    {}
    ~KGRotatedClosedPathSpace() override = default;

  public:
    std::shared_ptr<XPathType> Path()
    {
        return fPath;
    }
    const std::shared_ptr<XPathType> Path() const
    {
        return fPath;
    }

    void RotatedMeshCount(const unsigned int& aCount)
    {
        fRotatedMeshCount = aCount;
        return;
    }
    const unsigned int& RotatedMeshCount() const
    {
        return fRotatedMeshCount;
    }

  protected:
    mutable std::shared_ptr<XPathType> fPath;
    mutable unsigned int fRotatedMeshCount;

  public:
    void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const override
    {
        auto tJacket = std::make_shared<KGRotatedPathSurface<XPathType>>(fPath);
        tJacket->RotatedMeshCount(fRotatedMeshCount);
        tJacket->SetName("jacket");
        aBoundaryContainer.push_back(tJacket);

        return;
    }
    void VolumeAccept(KGVisitor* aVisitor) override
    {
        shapemsg_debug("rotated closed path volume named <" << GetName() << "> is receiving a visitor" << eom);
        auto* tRotatedClosedPathSpaceVisitor = dynamic_cast<typename KGRotatedClosedPathSpace::Visitor*>(aVisitor);
        if (tRotatedClosedPathSpaceVisitor != nullptr) {
            shapemsg_debug("rotated closed path volume named <" << GetName() << "> is accepting a visitor" << eom);
            tRotatedClosedPathSpaceVisitor->VisitRotatedClosedPathSpace(this);
            return;
        }
        KGVolume::VolumeAccept(aVisitor);
        return;
    }
    bool VolumeOutside(const katrin::KThreeVector& aPoint) const override
    {
        katrin::KTwoVector tZRPoint = aPoint.ProjectZR();
        return fPath->Above(tZRPoint);
    }
    katrin::KThreeVector VolumePoint(const katrin::KThreeVector& aPoint) const override
    {
        katrin::KTwoVector tZRPoint = aPoint.ProjectZR();
        katrin::KTwoVector tZRNearest = fPath->Point(tZRPoint);
        double tAngle = aPoint.AzimuthalAngle();
        return katrin::KThreeVector(cos(tAngle) * tZRNearest.R(), sin(tAngle) * tZRNearest.R(), tZRNearest.Z());
    }
    katrin::KThreeVector VolumeNormal(const katrin::KThreeVector& aPoint) const override
    {
        katrin::KTwoVector tZRPoint = aPoint.ProjectZR();
        katrin::KTwoVector tZRNormal = fPath->Normal(tZRPoint);
        double tAngle = aPoint.AzimuthalAngle();
        return katrin::KThreeVector(cos(tAngle) * tZRNormal.R(), sin(tAngle) * tZRNormal.R(), tZRNormal.Z());
    }

  private:
    static KGPlanarClosedPath* CompilerCheck()
    {
        XPathType* tPath = nullptr;
        return tPath;
    }
};

}  // namespace KGeoBag

#endif
