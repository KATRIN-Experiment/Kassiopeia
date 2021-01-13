#ifndef KGeoBag_KGPortHousingSurfaceMesher_hh_
#define KGeoBag_KGPortHousingSurfaceMesher_hh_

#include "KGComplexMesher.hh"
#include "KGPortHousingSurface.hh"

namespace KGeoBag
{
class KGPortHousingSurfaceMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGPortHousing>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGPortHousingSurfaceMesher() = default;
    ~KGPortHousingSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGPortHousing>* portHousingSurface) override;

    void ComputeEnclosingBoxLengths(std::vector<double>& theta, std::vector<double>& phi, std::vector<double>& mid,
                                    std::vector<double>& width);

    class PortDiscretizer
    {
      public:
        PortDiscretizer(KGPortHousingSurfaceMesher* d) : fPortHousingDiscretizer(d) {}

      protected:
        PortDiscretizer() : fPortHousingDiscretizer(nullptr) {}

        KGPortHousingSurfaceMesher* fPortHousingDiscretizer;
    };

    class RectangularPortDiscretizer : public KGPortHousingSurfaceMesher::PortDiscretizer
    {
      public:
        RectangularPortDiscretizer(KGPortHousingSurfaceMesher* d) : PortDiscretizer(d), fRectangularPort(nullptr) {}

        virtual ~RectangularPortDiscretizer() = default;

        void DiscretizePort(const KGPortHousing::RectangularPort* rectangularPort);

      private:
        RectangularPortDiscretizer() : PortDiscretizer(), fRectangularPort(nullptr) {}
        void PowerDistBoxCoord(int i, double length, double width, double* xyz);

        void PolygonBoxCoord(int i, double length, double width, double* xyz);

        void BoundingBoxCoord(int i, double length, double width, double* xyz);

        const KGPortHousing::RectangularPort* fRectangularPort;
    };

    class CircularPortDiscretizer : public KGPortHousingSurfaceMesher::PortDiscretizer
    {
      public:
        CircularPortDiscretizer(KGPortHousingSurfaceMesher* d) : PortDiscretizer(d), fCircularPort(nullptr) {}

        virtual ~CircularPortDiscretizer() = default;

        void DiscretizePort(const KGPortHousing::CircularPort* circularPort);

      private:
        CircularPortDiscretizer() : PortDiscretizer(), fCircularPort(nullptr) {}
        double Circle_theta(double r, int i);

        double Rect_theta(double r, int i);

        double Transition_theta(double r, int i);

        static void Circle_coord(double r, double theta, double p[3]);

        static void Rect_coord(double r, double theta, double p[3]);

        void Transition_coord(double r, double theta, double p[3]);

        const KGPortHousing::CircularPort* fCircularPort;
    };

  protected:
    static bool ChordsIntersect(double theta1min, double theta1max, double theta2min, double theta2max);
    static bool LengthsIntersect(double x1min, double x1max, double x2min, double x2max);

    std::shared_ptr<KGPortHousing> fPortHousing;
};
}  // namespace KGeoBag

#endif /* KGPORTHOUSINGSURFACEMESH_HH_ */
