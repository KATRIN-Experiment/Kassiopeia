#ifndef KGTORUSSURFACE_HH_
#define KGTORUSSURFACE_HH_

#include "KGRotatedCircleSurface.hh"

namespace KGeoBag
{

class KGTorusSurface : public KGRotatedCircleSurface
{
  public:
    KGTorusSurface();
    ~KGTorusSurface() override;

    static std::string Name()
    {
        return "torus_surface";
    }

  public:
    void Z(const double& aZ);
    void R(const double& anR);
    void Radius(const double& aRadius);
    void ToroidalMeshCount(const unsigned int& aToroidalMeshCount);
    void AxialMeshCount(const unsigned int& anAxialMeshCount);

    const double& Z() const;
    const double& R() const;
    const double& Radius() const;
    const unsigned int& ToroidalMeshCount() const;
    const unsigned int& AxialMeshCount() const;

  private:
    double fZ;
    double fR;
    double fRadius;
    unsigned int fToroidalMeshCount;
    unsigned int fAxialMeshCount;

  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitTorusSurface(KGTorusSurface* aTorusSurface) = 0;
    };

  public:
    void AreaInitialize() const override;
    void AreaAccept(KGVisitor* aVisitor) override;
};

}  // namespace KGeoBag

#endif
