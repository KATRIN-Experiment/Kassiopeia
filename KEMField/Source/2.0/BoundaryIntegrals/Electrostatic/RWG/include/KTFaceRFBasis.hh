#ifndef KTFACERFBASIS_HH_
#define KTFACERFBASIS_HH_

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "KElectrodeManager.hh"
#include "KTElectrode.hh"
#include "KTRobinHood.hh"
#include "KTBasis.hh"
#include "KTRWGFunctions.hh"
#include "KIOManager.hh"
#include "KMath.hh"

#include <complex>
#include <cmath>

namespace KEMField
{

/*!
  @class KTFaceRFBasis

  @author J. Formaggio

  @brief RobinHood FaceRF base class (for frequency-dependent applications)

  @details
  <b>Detailed Description:</b><br>
  Provides basis classes for RobinHood method to solve frequency-dependent boundary conditions.
  Uses Rao, Wilton, Glisson (FaceRF) common edge basis

  <b>Revision History:</b>
  \verbatim
  Date         Name          Brief description
  -----------------------------------------------
  21/03/2012   J. Formaggio     First version
  \endverbatim

 */

class KTFaceRFBasis : public KTBasis
{
public:

	//! Constructor
	/*!
		        Builds a basis class based on the FaceRF basis commonly used for frequency-dependent fields.
				This uses a face-base basis, and so the system requires the electrodes to be defined as
				KTElectrodes.
	 */
	KTFaceRFBasis();

	//! Destructor
	/*!
				Deletes factory and the static data that it stores.
	 */
	virtual ~KTFaceRFBasis();

	//! Returns the value of the vector element equivalent to eTarget
	/*!
					Returns a complex double value for this element
					/param eTarget Target electrode used for evaluation
	 */
	std::vector< Complex_t > GetScatteredElectric(double* r) const;
	std::vector< Complex_t > GetScatteredMagnetic(double* r) const;

    virtual Complex_t GetVectorElement(unsigned int iTarget) const;


	//! Returns the value of the matrix element equivalent to eTarget
	/*!
					Returns a complex double value for this element.
					Evuation carried from Source to Target
					/param eTarget Target electrode used for evaluation
					/param eSource Source electrode used for evaluation
	 */
	virtual Complex_t GetMatrixElement(unsigned int iTarget, unsigned int iSource) const;

	//! Returns the value of the solution element equivalent to eSource
	/*!
					Returns a complex double value for this element
					/param eSource Source electrode used for evaluation
	 */
	virtual Complex_t GetSolutionElement(unsigned int iSource) const;

	//! Sets the value of the solution to element eSource
	/*!
						/param eSource electrode where solution is stored
						/param aValue complex double value to be stored
	 */
	virtual void SetSolutionElement(unsigned int iSource, const Complex_t aValue)  const;

	//! Sets the value of the solution to element eSource
	/*!
						/param eSource electrode where solution is stored
						/param aValue  double value to be stored
	 */
	virtual void SetSolutionElement(unsigned int iSource, const double aValue)  const;

	//! Grabs the identification id for the group/electrode
	/*!
						/param index Identify the electrode's id number
	 */
	virtual int GetGroupID(unsigned int index)  const;

	//! Sets the identification id for the group/electrode
	/*!
						/param index  Identify the electrode's id number
						/param aValue Integer corresponding to group id number
	 */
	virtual void SetGroupID(unsigned int index, int aValue)  const;

	//! Locate a vector along the surface of the electrode
	std::vector<Complex_t> GetSurfaceVector(unsigned int iElement, const double* R = NULL) const;

	//! General dot product of A and B vectors
	template <typename SysVector>
	typename SysVector::value_type DotProduct(const SysVector &A, const SysVector &B) const
	{
		typename SysVector::value_type result;
		if (A.size() != B.size()) return result;
		for(unsigned int i = 0; i < A.size(); i++) result += A[i] * B[i];
		return result;
	};

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

	//! Kronecker delta function (for both integers and objects)
	template <typename SysObject>
	double KroneckerDelta(const SysObject &A, const SysObject &B)  const {return (A == B) ? 1. : 0.;};

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

protected:

	//!	Determine if these are magnetic or electric field components
	bool isElectric(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 0) ? true : false;}
	bool isMagnetic(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 1) ? true : false;}
	bool isMagneticTangent(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 2) ? true : false;}

	//!	Determine if these are induced charges/currents components
	bool isCharge(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 0) ? true : false;}
	bool isCurrent(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 1) ? true : false;}
	bool isCurrentTangent(unsigned int index) const {return ((unsigned int) (index/fNumElectrodes) == 2) ? true : false;}

	unsigned int GetElectrodeSubIndex(unsigned int index) const {return ((unsigned int) (index / fNumElectrodes) % fDegreeOfFreedom);}

	std::vector<Complex_t> GreenFunctionIntegral(const unsigned int iType, double* R, double k, unsigned int iElectrode) const;

	inline Complex_t ScaleFactor(unsigned int iType, KTElectrode* eTarget, unsigned int iSurface) const
	{
		Complex_t aValue = 0.;
		Complex_t s_j = (GetCoefficient(iType, eTarget, iSurface) * Normalization(iType,eTarget));
		if (norm(s_j) != 0.) aValue = 1./s_j;
		return aValue;
	}

	inline Complex_t Normalization(unsigned int iType, KTElectrode* eTarget) const
	{
		Complex_t aValue = 1.;
		Complex_t cU = 0.;
		for(unsigned int iSurface = 0; iSurface < 2; iSurface++){
			Complex_t c_j = GetCoefficient(iType, eTarget, iSurface);
			if (norm(c_j) != 0.) cU += pow(c_j, -2.);
		}
		if (norm(cU) != 0.) aValue = sqrt(cU);
		return aValue;
	}

	inline unsigned int GetType(unsigned int iTarget) const
	{
		unsigned int theType = 0;
		if (isCharge(iTarget)) theType = kElectricCharge;
		else theType = kElectricCurrent;
		return theType;
	};

	inline Complex_t GetCoefficient(unsigned int iType, KTElectrode* eTarget, unsigned int iSurface) const
	{
		Complex_t aValue = 0.;
		Complex_t eps = GetElectricPermittivity(eTarget, iSurface);
		Complex_t mu  = GetMagneticPermeability(eTarget, iSurface);

		if (iType == kElectricCharge  && norm(eps)>0.) aValue = sqrt(eps);
		if (iType == kMagneticCurrent && norm(eps)>0.) aValue = 1./sqrt(eps);
		if (iType == kElectricCurrent && norm(mu )>0.) aValue = 1./sqrt(mu);
		if (iType == kMagneticCharge  && norm(mu )>0.) aValue = sqrt(mu);

		return aValue;
	};
	enum kVariables {kElectricCharge, kMagneticCurrent, kElectricCurrent, kMagneticCharge};

	KEMFieldClassDef(KEMField::KTFaceRFBasis, 2);
};

} /* namespace KEMField */

#endif /* KTFACERFBASIS_HH_ */
