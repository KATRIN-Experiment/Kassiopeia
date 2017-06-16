// $Id$

/*
  Class: KTFaceRFBasis
  Author: J. Formaggio

  For full class documentation: KTFaceRFBasis.h

  Revision History
  Date         Name          Brief description
  -----------------------------------------------
  14/03/2012   J. Formaggio     First version

*/


#include "KTFaceRFBasis.hh"
#include "KIOManager.hh"

KEMFieldClassImp(KEMField::KTFaceRFBasis);

namespace KEMField
{

  KTFaceRFBasis::KTFaceRFBasis()
  {
    fDegreeOfFreedom = 3;		//  Has the option of being 3 degrees of freedom
    fNumElectrodes = 0;
    fFrequency = 0.;

  }

  KTFaceRFBasis::~KTFaceRFBasis()
  {
  }

  Complex_t KTFaceRFBasis::GetVectorElement(unsigned int iTarget) const
  {

    //!	This returns the basis function integrated over the incident electric and magnetic field

    Complex_t aValue (0., 0.);

    KTElectrode *eTarget;
    GetElectrode(iTarget, eTarget);

    double P[3];
    eTarget->Centroid(P, false);

    std::vector<Complex_t> theNormal(3);
    std::vector<Complex_t> theField(3);
    std::vector<Complex_t> theCurrent(3);

    for (unsigned int iSurface = 0; iSurface < 1; iSurface++){
      double k = GetWaveNumber(eTarget, iSurface);
      Complex_t sj = ScaleFactor(GetType(iTarget) , eTarget, iSurface);
      Complex_t scale = 0.;

      for(int j = 0; j < 3; j++) {
	theNormal[j] = pow(-1.,iSurface+1) * eTarget -> GetNormal(P,j);
      }

      theCurrent = GetSurfaceVector(iTarget, P);

      Complex_t theBoundaryCondition;
      if (isElectric(iTarget)) {
	theField = fExternalFields->GetExternalElectricField(P, k);
	theBoundaryCondition = DotProduct <std::vector<Complex_t> > (theNormal,theField);
	scale = sqrt(GetElectricPermittivity(eTarget,iSurface));
      }
      else
      {
	theField = fExternalFields->GetExternalMagneticField(P, k);
	theBoundaryCondition = DotProduct <std::vector<Complex_t> >(CrossProduct <std::vector<Complex_t> > (theNormal,theField), theCurrent);;
	scale = sqrt(GetMagneticPermeability(eTarget,iSurface));
      }
      aValue += theBoundaryCondition * scale * sj;
    }

    return aValue;

  }

  std::vector< Complex_t > KTFaceRFBasis::GetScatteredElectric(double* r) const {

    std::vector< Complex_t > aValue(3);

    for (unsigned int i = 0; i < GetNumElements(); i++){
      KTElectrode *eSource;
      GetElectrode(i, eSource);

      Complex_t im (0.,1);

      for(int iSurface = 0; iSurface < 2; iSurface++){

	Complex_t s_j = ScaleFactor(GetType(i) , eSource, iSurface);

	double k = GetWaveNumber(eSource, iSurface);
	Complex_t alpha = GetSolutionElement(i);

	Complex_t scale = sqrt(GetElectricPermittivity(eSource,iSurface));

	if (isCharge(i))
	{
	  std::vector< Complex_t > Nij(3);
	  Nij = GreenFunctionIntegral(kGradient, r, k, i);

	  if (norm(scale) != 0.){
	    for(int index = 0; index < 3; index++) aValue[index]  += (alpha) * (pow(-1.,iSurface+1) * s_j) * (Nij[index]) / scale;
	  }
	}
	else
	{
	  std::vector< Complex_t > Sij(3);
	  Sij = GreenFunctionIntegral(kVectorField, r, k, i);

	  if (norm(scale) != 0.){
	    for(int index = 0; index < 3; index++) aValue[index] += (alpha) * (pow(-1.,iSurface+1) * s_j) * (Sij[index]) * (-im * k) / scale;
	  }
	}
      }
    }
    return aValue;
  }

  std::vector< Complex_t > KTFaceRFBasis::GetScatteredMagnetic(double* r) const {

    std::vector< Complex_t > aValue(3);

    for (unsigned int i = 0; i < GetNumElements(); i++){
      KTElectrode *eSource;
      GetElectrode(i, eSource);

      Complex_t im (0.,1);
      for(int iSurface = 0; iSurface < 2; iSurface++){

	Complex_t s_j = ScaleFactor(kElectricCurrent, eSource, iSurface);
	double k = GetWaveNumber(eSource, iSurface);
	Complex_t alpha = GetSolutionElement(i);

	if (isCurrent(i) || isCurrentTangent(i))
	{
	  std::vector< Complex_t > Kij(3);
	  Kij = GreenFunctionIntegral(kCurl, r, k, i);
	  for(int index = 0; index < 3; index++) {
	    Complex_t scale = sqrt(GetMagneticPermeability(eSource,iSurface));
	    if (norm(scale) != 0.) aValue[index] -= (alpha) * (pow(-1.,iSurface+1) * s_j) * (Kij[index]) / scale;
	  }
	}
      }
    }
    return aValue;
  }

  Complex_t KTFaceRFBasis::GetMatrixElement(unsigned int iTarget, unsigned int iSource) const
  {

    //! Determines what values get filled in for the Green's function matrix elements
    //! Calculates field/potential from eSource to eTarget.

    Complex_t aValue (0., 0.);

    KTElectrode *eTarget;
    GetElectrode(iTarget, eTarget);

    KTElectrode *eSource;
    GetElectrode(iSource, eSource);

    double P[3];
    eTarget->Centroid(P, false);

    std::vector<Complex_t> Nij(3);
    std::vector<Complex_t> Kij(3);
    std::vector<Complex_t> Sij(3);
    std::vector<Complex_t> Jt(3);
    std::vector<Complex_t> Js(3);
    Complex_t Lij (0.,0.);

    for(int iSurface = 0; iSurface < 2; iSurface++){

      Complex_t im (0.,1);
      double k = GetWaveNumber(eTarget, iSurface);

      Complex_t s_i = ScaleFactor(GetType(iTarget), eTarget, iSurface);
      Complex_t s_j = ScaleFactor(GetType(iSource), eSource, iSurface);

      std::vector<Complex_t> theNormal(3);
      for(int j = 0; j < 3; j++) theNormal[j] = pow(-1.,iSurface+1) * eTarget -> GetNormal(P,j);

      if (isElectric(iTarget) && isCharge(iSource) ) {
	Complex_t Iij = KroneckerDelta <unsigned int> (iTarget, iSource) ;
	Nij = GreenFunctionIntegral(kGradient, P, k, iSource);

	if (KroneckerDelta <KTElectrode*> (eTarget,eSource) == 1.) Lij = 0.5 * Iij;
	else Lij += DotProduct <std::vector<Complex_t> > (theNormal,Nij);
      }

      if (isElectric(iTarget) && (!isCharge(iSource)) && (k > 0.)) {
	Sij = GreenFunctionIntegral(kVectorField, P, k, iSource);;
	Lij = -im * k * DotProduct <std::vector<Complex_t> >(theNormal,Sij);
      }

      if ((!isElectric(iTarget)) && (!isCharge(iSource))) {
	Complex_t Iij = KroneckerDelta <unsigned int> (iTarget, iSource);
	Kij = GreenFunctionIntegral(kCurl, P, k, iSource);

	Jt =GetSurfaceVector(iTarget, P);
	Js =GetSurfaceVector(iSource, P);

	if (KroneckerDelta <KTElectrode*> (eTarget,eSource) == 1.) Lij = 0.5 * DotProduct <std::vector<Complex_t> > (Jt,Js);
	else Lij += DotProduct <std::vector<Complex_t> >(CrossProduct <std::vector<Complex_t> > (theNormal,Kij), Jt);

      }
      aValue += (s_i * s_j) * Lij;
    }

    return aValue;

  }

  Complex_t KTFaceRFBasis::GetSolutionElement(unsigned int iSource) const
  {

    //! Returns charge values calculated at end of solver method

    Complex_t aValue (0.,0.);

    KTElectrode *eSource;
    GetElectrode(iSource, eSource);

    if (isCharge(iSource)) aValue = eSource->GetElectricCurrentWeight(0);
    if (isCurrent(iSource)) aValue = eSource->GetElectricCurrentWeight(1);
    if (isCurrentTangent(iSource)) aValue = eSource->GetElectricCurrentWeight(2);

    return aValue;

  }

  void KTFaceRFBasis::SetSolutionElement(unsigned int iSource, const Complex_t aValue) const
  {
    //! Fills in charge values calculated at end of RobinHood Method

    KTElectrode *eSource;
    GetElectrode(iSource, eSource);

    if (isCharge(iSource)) eSource->SetElectricCurrentWeight(0,aValue);
    if (isCurrent(iSource)) eSource->SetElectricCurrentWeight(1,aValue);
    if (isCurrentTangent(iSource)) eSource->SetElectricCurrentWeight(2,aValue);

  }

  void KTFaceRFBasis::SetSolutionElement(unsigned int iSource, const double aValue) const
  {
    //! Fills in charge values calculated at end of RobinHood Method

    Complex_t aValueComplex (aValue,0.);
    SetSolutionElement(iSource, aValueComplex);

  }

  int KTFaceRFBasis::GetGroupID(unsigned int index)  const
  {
    KTElectrode *eElectrode;
    GetElectrode(index, eElectrode);

    return eElectrode -> GetPhysicalID();
  }

  void KTFaceRFBasis::SetGroupID(unsigned int index, int aValue)  const
  {
    KTElectrode *eElectrode;
    GetElectrode(index, eElectrode);

    eElectrode -> SetPhysicalID(aValue);
  }

  std::vector<Complex_t> KTFaceRFBasis::GetSurfaceVector(unsigned int iTarget, const double* P) const
  {
    std::vector<Complex_t> aValue(3);

    unsigned int index = 0;
    if (isCurrentTangent(iTarget) || isMagneticTangent(iTarget)) index = 1;

    KTElectrode *theTarget;
    GetElectrode(iTarget, theTarget);

    double Q[3];
    theTarget -> GetVertex(index,Q);

    double C[3];
    theTarget -> Centroid(C, false);

    for(unsigned int i = 0; i < 3; i++) aValue[i] = (Q[i] - C[i]);

    return aValue;
  }

  std::vector<Complex_t> KTFaceRFBasis::GreenFunctionIntegral(const unsigned int iType, double* R, double k, unsigned int iElectrode) const
  {
    std::vector<Complex_t> theResult(3);

    KTElectrode* theSource;
    GetElectrode(iElectrode, theSource);

    KTPairElectrode* theElement = dynamic_cast<KTPairElectrode*> (theSource);
    if (theElement == NULL)
    {
      double pPoint[3];

      unsigned int jindex = 0;
      if (isCurrentTangent(iElectrode)) jindex = 1;
      theSource -> GetVertex(jindex,pPoint);

      KTRWGFunctions* fRWGFunc = new KTRWGFunctions(theSource,R,pPoint);
      fRWGFunc -> InitVectors();

      if (iType == kCurl){
	theResult = fRWGFunc->CalcGreenFunction(k,kGradient);
	theResult = CrossProduct < std::vector<Complex_t> > (theResult, GetSurfaceVector(iElectrode,R));
      } else if (iType == kVectorField){
	std::vector<Complex_t> theCurrent(3);
	theCurrent = GetSurfaceVector(iElectrode,R);
	theResult = fRWGFunc->CalcGreenFunction(k,kScalar);
	for(unsigned int j = 0; j < 3; j++) theResult[j] = theResult[0] * theCurrent[j];
      } else {
	theResult = fRWGFunc->CalcGreenFunction(k,iType);
      }

      delete fRWGFunc;
    }
    else
    {
      for (int iElement = 0; iElement < (int) theSource->GetNSubElements(); iElement++)
      {
	KTElectrode* theSubElement = theElement->GetElectrode(iElement);

	double pPoint[3];
	theSubElement -> GetSourcePoints(iElement,pPoint);
	KTRWGFunctions* fRWGFunc = new KTRWGFunctions(theSubElement,R,pPoint);
	fRWGFunc -> InitVectors();

	std::vector<Complex_t> F = fRWGFunc->CalcGreenFunction(k,iType);
	for(int j = 0; j < 3; j++) theResult[j] += pow(-1.,iElement+1) * F[j];

	delete fRWGFunc;
      }
    }
    return theResult;

  }


} /* namespace KEMField */
