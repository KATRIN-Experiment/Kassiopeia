#ifndef KTRWGFUNCTIONS_HH_
#define KTRWGFUNCTIONS_HH_

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "KElectrodeManager.hh"
#include "KMath.hh"
#include "TVector3.h"
#include "KTElectrode.hh"
#include "KTPairElectrode.hh"
#include "KTRobinHood.hh"
#include "KTBasis.hh"
#include "KIOManager.hh"

#include "KVMFieldWrapper.hh"
#include "KVMElectrodeIntegrator.hh"

#include <complex>
#include <cmath>

namespace KEMField
{

/*!
  @class KTRWGFunctions

  @author J. Formaggio

  @brief RobinHood RWG function base class (for RF applications)

  @details
  <b>Detailed Description:</b><br>
  Provides functions for basis classes for RobinHood method.
  Uses Rao, Wilton, Glisson (RWG) common edge basis

  <b>Revision History:</b>
  \verbatim
  Date         Name          Brief description
  -----------------------------------------------
  21/03/2012   J. Formaggio     First version
  \endverbatim

 */

class KTRWGFunctions
{
public:

	//! Constructor
	/*!

		Creates frequency-dependent Green Functions evaluated over a surface.
	 */
	KTRWGFunctions();

	//! Constructor
	/*!

		Creates frequency-dependent Green Functions evaluated over a surface.
		/param eSource Source electrode to be integrated
		/param pPosition Distance from point of evaluation to eSource
		/param pSource Vertex of polygon from which gradient is computed
	 */
	KTRWGFunctions(KTElectrode *eSource, double* pPosition, double* pSource);

	//! Destructor
	/*!
				Deletes factory and the static data that it stores.
	 */
	virtual ~KTRWGFunctions();

	//! Clear all vectors
	void Clear();

	//! Initialize all vectors to be used in calculation
	void InitVectors();

	//! Sets Electrode over which computation is perfomred.
	/*!
	 	 /param eSource Electrode to be inserted in system
	 */
	void SetElectrode(KTElectrode *eSource);

	//! Returns electrode used in calculation
	KTElectrode* GetElectrode() {return fElectrode;}

	//! Sets position over which computation is performed.
	/*!
	 	 /param pPosition Distance from point of evaluation to eSource
	 */
	void SetPosition(double* pPosition);

	//! Returns 3-vector for position
	TVector3 GetPosition() {return fPosition;}

	//! Sets source point over which computation is performed.
	/*!
	 	 /param pSource Vertex of polygon from which gradient is computed
	 */
	void SetSourcePoint(double* pSource);

	//! Returns 3-vector source position
	TVector3 GetSourcePoint() {return fSourcePoint;}

	//! Calculate solid angle that is used in calculation
	void CalcSolidAngle();

	//! Return solid angle that is used in calculation
	double GetSolidAngle(){ return fSolidAngle;}

	//! Returns Euler angles for spherical polygon
	double CalcEulerAngles(unsigned int pIndex, unsigned int qIndex);

	//! Set accuracy scale over which expansion should stop
	//	/param aValue Value of accuracy
	void SetAccuracy(const double aValue) {fAccuracy = aValue;}

	//! Return accuracy setting
	double GetAccuracy() {return fAccuracy;}

	//! Evaluate line integral of Green's function (R^q)
	/*!
		/param qIndex Power of |r-r'| index
		/param iStart Which vertex to start with
		/param iEnd   Which vertex to end (evaluate line integral from iStart to iEnd)
	*/
	double FunctionIqL(int qIndex, unsigned int iStart, unsigned int iEnd);

	//! Evaluate surface integral of Green's function (R^q)
		/*!
			/param qIndex Power of |r-r'| index
		*/
	double FunctionIqS(int qIndex);

	//! Return static potential limit
	double Potential(double* pPosition, bool sym = false, double k = 0.);

	//! Return static electric field limit
	void EField(double *pPosition,double *field,bool sym = false, double k = 0.);

	//! Return area as calculated by integral
	double GetAreaFromIntegral();

	//! Return area of vector field as calculated by integral
	std::vector<std::complex <double> > GetVectorIntegral();

	//! Store values of surface integrals
	void SetIqS(int index, double aValue);

	//! Determine if surface integral has already been computed (and return value at aValue)
	bool GetIqS(int index, double& aValue) const;

	//! Store values of surface integrals
	void SetIqL(int index, unsigned int iStart, unsigned int iEnd, double aValue);

	//! Determine if surface integral has already been computed (and return value at aValue)
	bool GetIqL(int index, unsigned int iStart, unsigned int iEnd, double& aValue) const;

	//! Get maximum number of terms in expansion
	int GetQMax() const {return fQmax;}

	//! Set maximum number of terms in expansion
	void  SetQMax(const int aValue) {fQmax = aValue;}

	//! Get distance magnitude
	double GetDistance() {return fDeltaR.Mag();}

	//! Perform Green Function calculation
	/*!
	 	 /param k  Wavenumber (meter^-1) used for calculation
	 	 /param iType Type of calculation
	 	  * iType = 1    G(r,r') div(f(r') dS'
	 	  * iType = 2    G(r,r') f(r') dS'
	 	  * iType = 3    used in iType = 4
	 	  * IType = 4    Grad(G(r,r')) x f(r') dS'
	 	  Where Grad is gradient, G(r,r') is the Green's function, f(r') is the basis function, and dS' is the surface area.
	 	  r is the evaluation point, and r' is the dummy variable
	 */
	std::vector<Complex_t> CalcGreenFunction(const double k, const unsigned int iType);

	std::vector<Complex_t> GetNumericalIntegral(const double k, const unsigned int iType, bool isSmooth = false);

	std::vector<Complex_t> GetSmoothIntegral(const double k, const unsigned int iType);

protected:

	//!	Functions that determine if these are induced charges/currents from magnetic or electric field components

	//! Electrode used in surface integral calculation
	KTElectrode *fElectrode;

	//!  Number of verticies in polygon
	unsigned int fNVerticies;

	//!  Maximum number of terms in expansion of R^q
	int fQmax;

	//!  Minimum number of terms in expansion of R^q
	int fQmin;

	//!  Position of r, where function is being evaluated
	TVector3 fPosition;

	//!  Center of surface
	TVector3 fCenter;

	//!  3-vector distance from center of electrode to evaluation point (r-r')
	TVector3 fDeltaR;

	//!  Normal vector to surface
	TVector3 fNormal;

	//!  Internal vector
	TVector3 fRho;

	//!  Free vertex in RWG basis
	TVector3 fSourcePoint;

	//!  Polygon ordering of verticies
	std::vector <unsigned int> fOrder;

	//!  Polygon vertices
	std::vector <TVector3> fVertex;

	//!  Vectors outward from side of polygon in the polygon plane
	std::vector <TVector3> fOutward ;

	//!  Vectors along side of polygon in the polygon plane
	std::vector <TVector3> fAlongSide ;

	//!  Vectors at edge of polygon in the polygon plane
	std::vector <TVector3> fEdge ;

	//!  Internal variable
	std::vector <TVector3> a_n;

	//!  Storing of surface and line integrals

	std::map<int,double> fIqS_Map;
	std::map<int,double> fIqL_Map;

	//!  Solid angle (shadow) of trangle
	double fSolidAngle;

	//! distance from triangle plane to r
	double fH;

	//!  Value of accuracy
	double fAccuracy;

    enum kGreenFunction {kNull, kScalar, kVectorField, kGradient, kCurl, kCurlScalar, kCurlVector, kCurlNormal, kCurlTangent, kSelfTerm};

    KEMFieldClassDef(KEMField::KTRWGFunctions, 2);
};

class FieldNumerical
{
    public:
		FieldNumerical(){isSmooth = false;};
        virtual ~FieldNumerical(){;};

    	//! Locate a vector along the surface of the electrode
		std::vector<Complex_t> GetSurfaceVector(KTElectrode *theTarget, const double* P, unsigned int index = 0) const
		{
			std::vector<Complex_t> aValue(3);

			double Q[3];
			theTarget -> GetSourcePoints(index,Q);

			double C[3];
			theTarget -> Centroid(C, false);

			Complex_t length = 0.;
			for(unsigned int i = 0; i < 3; i++) {
				aValue[i] = Q[i] - P[i];
				length += pow(Q[i] - C[i],2.);
			}
			for(unsigned int i = 0; i < 3; i++) aValue[i] /= sqrt(length);

			return aValue;
		}


		Complex_t GreenFunction(const double* in) const
		{
			Complex_t theResult;

        	double R = dist(in);
        	double k = fWaveNumber;
        	Complex_t ikR (0.,k * R);
        	theResult = (1./(4. * TMath::Pi()) * (exp(ikR)/R));

        	if (isSmooth) theResult += 1./(4. * TMath::Pi()) * (-1./R + k*k*R/2.);

        	return theResult;
		}

		std::vector <Complex_t> GradientGreenFunction(const double* in) const
		{
			std::vector <Complex_t> theResult(3);

        	double r[3];
        	radius_vec(in,r);

        	double R = dist(in);
        	double k = fWaveNumber;
        	Complex_t ikR (0.,k * R);
        	for(unsigned int j = 0; j < 3; j++) {
        		theResult[j] = 1./(4. * TMath::Pi()) * (r[j] * (ikR - 1.) * exp(ikR)/pow(R,3.));
            	if (isSmooth) theResult[j] += 1./(4. * TMath::Pi()) * (r[j]/R) * (+1./pow(R,2.) + k*k/2.);
        	}
        	return theResult;
		}

		//function which takes a point in 3d (x,y,z) and returns the green function e^ikr/r
        void GetScalar(const double* in, double* out) const
        {
        	out[0] = real(GreenFunction(in));
        	out[1] = imag(GreenFunction(in));
        }

        void GetVectorField(const double* in, double* out) const
        {
        	std::vector<Complex_t> J = GetSurfaceVector(fElectrode, in);
        	for(unsigned int j = 0; j < 3; j++) {
        		out[2*j]   = real(J[j] * GreenFunction(in));
        		out[2*j+1] = imag(J[j] * GreenFunction(in));
        	}
        }

        //function which takes a point in 3d (x,y,z) and returns the gradient of the green function e^ikr/r
        void GetGradient(const double* in, double* out) const
        {
        	std::vector <Complex_t> G = GradientGreenFunction(in);
        	for(unsigned int j = 0; j < 3; j++) {
        		out[2*j]   = real(G[j]);
        		out[2*j+1] = imag(G[j]);
        	}
        }

        void GetCurl(const double* in, double* out) const
        {

        	std::vector<Complex_t> J = GetSurfaceVector(fElectrode, in);

        	std::vector<Complex_t> F(3);
        	std::vector <Complex_t> G = GradientGreenFunction(in);
        	F = CrossProduct < std::vector<Complex_t> > (G,J);
           	for(unsigned int j = 0; j < 3; j++) {
           		out[2*j]   = real(F[j]);
            	out[2*j+1] = imag(F[j]);
           	}
        }

        double dist(const double* xyz) const
        {
            //compute distance to origin
            double dx,dy,dz;
            dx = xyz[0] - fOrigin[0];
            dy = xyz[1] - fOrigin[1];
            dz = xyz[2] - fOrigin[2];
            return sqrt(dx*dx + dy*dy + dz*dz);
        };

        void radius_vec(const double* dom, double* range) const
        {
            range[0] = dom[0] - fOrigin[0];
            range[1] = dom[1] - fOrigin[1];
            range[2] = dom[2] - fOrigin[2];
        }

        void SetOrigin(TVector3 origin){fOrigin = origin;};

        void SetWaveNumber(double k) {fWaveNumber = k;};

        void SetElectrode(KTElectrode *e) {fElectrode = e;};

        void SetSmooth(bool aValue) {isSmooth = aValue;};

        std::vector<Complex_t> GetSurfaceVector(const double* P, unsigned int index) const
        {
        	std::vector<Complex_t> aValue(3);

        	double Q[3];
        	fElectrode -> GetSourcePoints(index,Q);

        	double C[3];
        	fElectrode -> Centroid(C, false);

        	Complex_t length = 0.;
        	for(unsigned int i = 0; i < 3; i++) {
        		aValue[i] = Q[i] - P[i];
        		length += pow(Q[i] - C[i],2.);
        	}
        	for(unsigned int i = 0; i < 3; i++) aValue[i] /= 2.*sqrt(length);

        	return aValue;
        }
    	//! General cross product of A and B vectors in 3-dimensions
    	template <typename SysVector>
    	SysVector CrossProduct(const SysVector &A, const SysVector &B) const
    	{
    		const unsigned int nDim = A.size();
    		SysVector result(nDim);

    		if (nDim != B.size() || nDim !=3) return result;

    		for(int i = 0; i < nDim; i++){
    			for(int j = 0; j < nDim; j++){
    				for(int k = 0; k < nDim; k++){
    					result[i] += LeviCivita(i,j,k) * A[j] * B[k];
    				}
    			}
    		}
    		return result;
    	};

    	//! Levi-Civita function (3 dimensions only)
    	double LeviCivita(const int i, const int j, const int k) const {
    		double result = 0.;
    		if ((i < 3) && (j < 3) && (k < 3))
    		{
    			result = (double) ((i - j) * (j - k) * (k - i));
    			result /=2.;
    		}
    		return result;
    	};

    private:

    	bool isSmooth;
    	TVector3 fOrigin;
        double fWaveNumber;
        KTElectrode *fElectrode;

};

} /* namespace KEMField */

#endif /* KTRWGBASIS_HH_ */
