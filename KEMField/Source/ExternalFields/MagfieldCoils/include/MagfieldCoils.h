//         MagfieldCoils.h:  3-dim magnetic field calculation of coils

#ifndef MagfieldCoils_h
#define MagfieldCoils_h

#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;


////////////////////////////////////

class MagfieldCoils
{
  public:
      MagfieldCoils(string inputdirname, string inputobjectname, string inputcoilfilename,
		               int inputNelliptic, int inputnmax, double inputepstol);   // constructor
      MagfieldCoils(string inputdirname, string inputobjectname);
     ~MagfieldCoils();  // deconstructor
     bool Magfield(const double *P, double *B);
     void MagfieldElliptic(const double *P, double *B);
     void SetTrackingStart();
  private:
// MEMBER FUNCTIONS:
// General:
     void CoilRead();
     void CoilGroupWrite();
     void CoilGroupRead();
     void MagsourceRead();
     void DynamicMemoryAllocations();
// Elliptic:
     void Magfield2EllipticCoil(int i, double z, double r, double& Bz,double& Br);
     void MagfieldEllipticCoil(int i, const double *P, double *B);
// Source Remote:
     double Funrorem(int i, double z0);
     void Magsource2RemoteCoil(int i, double z0, double rorem);
     void MagsourceRemoteCoils();
     void MagsourceRemoteGroups();
     void RemoteSourcepointGroup(int g);
// Source Magcharge:
    void MagsourceMagchargeCoils();
// Source Central:
     double Funrocen(int i, double z0);
     double FunrocenG(int g, double z0);
     void Magsource2CentralCoil(int i, double z0, double rocen);
     void MagsourceCentralCoils();
     void CentralSourcepointsCoil(int i);
     void MagsourceCentralGroups();
     void CentralSourcepointsGroup(int g);
// Magfield Remote:
    bool Magfield2Remote(bool coil, int ig, double z, double r, double& Bz,double& Br,double& rc);
// Magfield Magcharge:
    bool Magfield2Magcharge(int i, double z, double r, double& Bz, double& Br, double& rc);
    bool Hfield(int i, double z, double r, double& Hz, double& Hr, double& rc);
// Magfield Central:
    bool Magfield2Central(bool coil, int ig, int j, double z, double r, double& Bz,double& Br,double& rc);
// Magfield Coil, Group:
    bool MagfieldCoil(int i, const double *P, double *B);
    bool MagfieldGroup(int g, const double *P, double *B);
    bool SourcePointSearching(bool coil, int ig, double z, double r, int type, int& jbest, double& rcbest);
// Integration:
     void GaussLegendreIntegration(int& N, double a, double b, double* x, double* w);
// Carlson elliptic:
     double RF_Carlson(double x,double y,double z);
     double RD_Carlson(double x,double y,double z);
     double RJ_Carlson(double x,double y,double z,double p);
     double RC_Carlson(double x,double y);
// Simple:
     double pow2(double x);
     double FMIN(double a, double b);
     double FMAX(double a, double b);
     double FMIN3(double a, double b, double c);
     double FMAX3(double a, double b, double c);
     double Mag(double* vec);
     double Mag(double x, double y, double z);
// VARIABLES:
     int Ncoil; // number of coils; coil indexing: i=0, ..., Ncoil-1
     double** coil; // coil parameters: coil[i][j], i=0, ..., Ncoil-1; j=0,...,13.
     string dirname; // name of directory where the coil and source coefficient data files are stored
     string coilfilename; // name of coil input file
     string objectname; // this string identifies the Magfield object (used as starting part of the source files)
     int Nelliptic; // radial numerical integration parameter for magnetic field calc. with elliptic integrals
     int nmax; // number of source coefficients for fixed z: nmax+1
     double epstol; // small tolerance parameter needed for the symmetry group definitions (distance: in meter)
     int Ng; // number of coil symmetry groups; group indexing:  g=0, ..., Ng-1
     int* G; // coil with global index i is in symmetry group g=G[i], i=0,...,Ncoil-1
     int* Nc; // number of coils in symmetry group g: Nc[g]; local coil indexing in group g: c=0,..., Nc[g]-1
     int** Cin; // global coil indices i in symmetry group g: i=Cin[g][c],...,c=0 to Nc[g]-1  (g=0,...,Ng-1)
     double** Line; // symmetry group axis line of symmetry group g (g=0,...,Ng-1):
//      line point:  Line[g][0], Line[g][1], Line[g][2];  line direction unit vector:  Line[g][3], Line[g][4], Line[g][5];
     double** Z; // local z coordinate of coil center in symmetry group g:  Z[g][c],   c=0,..., Nc[g]-1;   Z[g][0]=0
                       //   (g: group index,  c: local coil index)
     double *C0, *C1, *c1, *c2, *c3, *c4, *c5, *c6, *c7, *c8, *c9, *c10, *c11, *c12;

     double* Pp, *P; // Legendre polynomial P and first derivative Pp
     double* Bzplus, *Brplus; // used in Bz and Br mag. field calc.
// Remote source constant variables:
     double* Brem1; // 1-dim. remote source constants Brem1[n], n=2,...,nmax.
     double* rorem; // remote convergence radius for coil i: rorem[i]
     double** Brem; // remote source constant for coil i: Brem[i][n] (coil index i, source constant index n)
     double* z0remG; // remote source point for group G: z0remG[g]
     double* roremG; // remote convergence radius for group G: roremG[g]
     double** BremG; // remote source constant for symmetry group g: BremG[g][n] (group index g, source constant index n)
// Magnetic charge source constant variables:
     double** Vrem; // magnetic charge remote source constant for coil i: Vrem[i][n] (coil index i, source const. index n)
// Central source constant variables:
     double* Bcen1; // 1-dim. central source constants Bcen1[n], n=0,...,nmax.
     int* Nsp; // number of central source points for coil i: Nsp[i] (i=0,...,Ncoil-1)
     double** z0cen; // central source point local z value for coil i: z0cen[i][j] (coil index i, source point index j)
     double** rocen; // central convergence radius for coil i: rocen[i][j] (coil index i, source point index j)
     double*** Bcen; // central source constant for coil i: Bcen[i][j][n] (coil index i, source point index j, source constant index n)
     int* NspG; // number of central source points for group g: Nsp[g] (g=0,...,Ng-1)
     double** z0cenG; // central source point local z value: z0cenG[g][j] (group index g, source point index j)
     double** rocenG; // central convergence radius: rocenG[g][j] (group index g, source point index j)
     double*** BcenG; // central source constant: BcenG[g][j][n] (group index g, source point index j, source const. index n)
     int *jlast, *jlastG;  // last central source point index for coil and group calculation
double rclimit;
};


////////////////////////////////////////////////////////

inline double MagfieldCoils::pow2(double x)
{
   return x*x;
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::FMIN(double a, double b)
{
   return ((a)<=(b)?(a):(b));
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::FMAX(double a, double b)
{
   return ((a)>(b)?(a):(b));
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::FMIN3(double a, double b, double c)
{
   return (FMIN(a,b)<=(c)?(FMIN(a,b)):(c));
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::FMAX3(double a, double b, double c)
{
   return (FMAX(a,b)>(c)?(FMAX(a,b)):(c));
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::Mag(double* vec)
{
   return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}

////////////////////////////////////////////////////////

inline double MagfieldCoils::Mag(double x, double y, double z)
{
   return sqrt(x*x+y*y+z*z);
}

////////////////////////////////////////////////////////

inline void MagfieldCoils::SetTrackingStart()
{
      for(int i=0; i<Ncoil; i++)
         jlast[i]=-1;
      for(int g=0; g<Ng; g++)
         jlastG[g]=-1;
}


#endif
