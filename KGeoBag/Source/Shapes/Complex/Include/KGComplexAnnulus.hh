#ifndef KGComplexAnnulus_DEF
#define KGComplexAnnulus_DEF

#include <vector>

#include "KGCoordinateTransform.hh"

namespace KGeoBag
{
  class KGComplexAnnulus
  {
  public:
    KGComplexAnnulus() : fCoordTransform(NULL) {}
    KGComplexAnnulus(double rmain);

    virtual ~KGComplexAnnulus();

    static std::string Name() { return "complex_annulus"; }

    void AddRing(double ASub[2],
			 double rsub);

    virtual KGComplexAnnulus* Clone() const;

    virtual void Initialize() const;

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,double* P_in=NULL,double* P_norm=NULL) const;

    void SetRMain(double r)     {fRMain = r; }
    void SetRadialMeshMain(int i) const { fRadialMeshMain = i; }
    void SetPolyMain(int i) const { fPolyMain = i; }

    double   GetRMain()        const { return fRMain; }
    int      GetRadialMeshMain()  const { return fRadialMeshMain; }
    int      GetPolyMain()     const { return fPolyMain; }

    const KGCoordinateTransform* GetCoordinateTransform() const { return const_cast<const KGCoordinateTransform*>(fCoordTransform); }

    class Ring
    {
    public:
      Ring() {}
      Ring(KGComplexAnnulus* complexAnnulus) :fComplexAnnulus(complexAnnulus) {}
      Ring(KGComplexAnnulus* complexAnnulus,
		   double       ZSub[2],
		   double       rsub);

      virtual ~Ring();

      virtual Ring* Clone(KGComplexAnnulus*) const;

      virtual void Initialize();

      KGComplexAnnulus* GetComplexAnnulus() const { return fComplexAnnulus; }
      void SetComplexAnnulus(KGComplexAnnulus* a) { fComplexAnnulus = a; }

      virtual void ComputeLocalFrame(double *cen) const;

      virtual bool   ContainsPoint(const double* P) const;
      double DistanceTo(const double* P,double* P_in=NULL,double* P_norm=NULL) const;

      const KGCoordinateTransform* GetCoordinateTransform() const { return const_cast<const KGCoordinateTransform*>(fCoordTransform); }

      // intrinsic characteristics of the port:
      void SetASub(double d[2]) { for (int i=0;i<2;i++) fASub[i] = d[i]; }
      void SetRSub(double d)    { fRSub = d; }

      double GetASub(int i)  const { return (i<2) ? fASub[i] : 0; }
      double GetRSub()         const { return fRSub; }
      double GetNorm(int i)  const { return (i<2) ? fNorm[i] : 0; }

      // discretization parameters
      void SetRadialMeshSub(int i)   const { fRadialMeshSub = i; }
      void SetPolySub(int i)      const { fPolySub = i; }

      int    GetRadialMeshSub()   const { return fRadialMeshSub; }
      int    GetPolySub()      const { return fPolySub; }

    private:

      // point describing the free end of the subordinate cyl.
      double fASub[3];
      // radius of the subordinate cylinder
      double fRSub;

      // unit vector (global coords) pointing from fASub to fCen
      double fNorm[3];

      // Determines the number of subelements created from the subordinate
      // cylinder during discretization
      mutable int fRadialMeshSub;

      // the # of sides of the polygon formed by the cross-section of the
      // subordinate cylinder when it is discretized
      mutable int fPolySub;

    protected:

      // pointer to port that owns this valve
      KGComplexAnnulus* fComplexAnnulus;
      // This could be changed to be handled by an external library
      KGCoordinateTransform* fCoordTransform;

      // local origin and x,y,z unit vectors in the global frame. Local Origins don't tilt, so normal unit vectors are fine, just center gets shifted.
      double fCen[3];
      double fX_loc[3] = {1,0,0};
      double fY_loc[3] = {0,1,0};
      double fZ_loc[3] = {0,0,1};
    };

    void AddRing(KGComplexAnnulus::Ring*);

    unsigned int GetNRings() const { return fRings.size(); }
    const KGComplexAnnulus::Ring* GetRing(int i) const { return const_cast<const KGComplexAnnulus::Ring*>(fRings.at(i)); }

  private:
    // radius of the main ring
    double fRMain;

    // Determines the number of subelements created from the main ring
    // during discretization
    mutable int fRadialMeshMain;

    // the # of sides of the polygon formed by the cross-section of the main
    // ring when it is discretized
    mutable int fPolyMain;

    // vector of sub rings associated with this port
    std::vector<KGComplexAnnulus::Ring*> fRings;

    // This could be changed to be handled by an external library
    mutable KGCoordinateTransform* fCoordTransform;
  };

}

#endif /* ComplexAnnulus_DEF */
