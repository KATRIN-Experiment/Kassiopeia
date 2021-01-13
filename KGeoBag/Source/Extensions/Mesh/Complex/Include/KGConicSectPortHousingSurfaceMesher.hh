#ifndef KGeoBag_KGConicSectPortHousingSurfaceMesher_hh_
#define KGeoBag_KGConicSectPortHousingSurfaceMesher_hh_

#include "KGComplexMesher.hh"
#include "KGConicSectPortHousingSurface.hh"

namespace KGeoBag
{
class KGConicSectPortHousingSurfaceMesher :
    virtual public KGComplexMesher,
    public KGConicSectPortHousingSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGConicSectPortHousingSurfaceMesher() = default;
    ~KGConicSectPortHousingSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGConicSectPortHousingSurface* conicSectPortHousingSurface) override;

    void ComputeEnclosingBoxDimensions(std::vector<double>& z_mid, std::vector<double>& r_mid,
                                       std::vector<double>& theta, std::vector<double>& z_length,
                                       std::vector<double>& alpha);

    class PortDiscretizer
    {
      public:
        PortDiscretizer(KGConicSectPortHousingSurfaceMesher* d) : fConicSectPortHousingDiscretizer(d) {}

      protected:
        PortDiscretizer() : fConicSectPortHousingDiscretizer(nullptr) {}
        KGConicSectPortHousingSurfaceMesher* fConicSectPortHousingDiscretizer;
    };

    class ParaxialPortDiscretizer : public KGConicSectPortHousingSurfaceMesher::PortDiscretizer
    {
      public:
        ParaxialPortDiscretizer(KGConicSectPortHousingSurfaceMesher* d) : PortDiscretizer(d), fParaxialPort(nullptr) {}

        virtual ~ParaxialPortDiscretizer() = default;

        void DiscretizePort(const KGConicSectPortHousing::ParaxialPort* paraxialPort);

      private:
        ParaxialPortDiscretizer() : PortDiscretizer(), fParaxialPort(nullptr) {}
        void Circle_coord(int i, double /*r*/, double p[3]);

        void Fan_coord(int i, double /*r*/, double p[3]);

        void Transition_coord(int i, double r, double p[3]);

        const KGConicSectPortHousing::ParaxialPort* fParaxialPort;
    };

    class OrthogonalPortDiscretizer : public KGConicSectPortHousingSurfaceMesher::PortDiscretizer
    {
      public:
        OrthogonalPortDiscretizer(KGConicSectPortHousingSurfaceMesher* d) : PortDiscretizer(d), fOrthogonalPort(nullptr)
        {}

        virtual ~OrthogonalPortDiscretizer() = default;

        void DiscretizePort(const KGConicSectPortHousing::OrthogonalPort* orthogonalPort);

      private:
        OrthogonalPortDiscretizer() : PortDiscretizer(), fOrthogonalPort(nullptr) {}
        static void Circle_coord(int i, double /*r*/, double p[3], std::vector<double>& x_int,
                                 std::vector<double>& y_int, std::vector<double>& z_int);

        void Fan_coord(int i, double /*r*/, double p[3]);

        void Transition_coord(int i, double r, double p[3], std::vector<double>& x_int, std::vector<double>& y_int,
                              std::vector<double>& z_int);

        const KGConicSectPortHousing::OrthogonalPort* fOrthogonalPort;
    };

  protected:
    static bool ChordsIntersect(double theta1min, double theta1max, double theta2min, double theta2max);

    static bool LengthsIntersect(double x1min, double x1max, double x2min, double x2max);

    std::shared_ptr<KGConicSectPortHousing> fConicSectPortHousing;
};
}  // namespace KGeoBag

#endif /* KGCONICSECTPORTHOUSINGMESHER_HH_ */
