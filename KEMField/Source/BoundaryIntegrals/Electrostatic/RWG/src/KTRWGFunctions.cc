// $Id$

/*
  Class: KTRWGFunctions
  Author: J. Formaggio

  For full class documentation: KTRWGBasis.h

  Revision History
  Date         Name          Brief description
  -----------------------------------------------
  25/03/2012   J. Formaggio     First version
  26/04/2012   J. Formaggio     Generalized to n-vertex polygons

 */


#include "KTRWGFunctions.hh"

#include "KTElectrode.hh"


KTRWGFunctions::KTRWGFunctions() :
    fElectrode(NULL),
    fNVerticies(0),
    fQmax(8),
    fQmin(4),
    fPosition(),
    fCenter(),
    fDeltaR(),
    fNormal(),
    fRho(),
    fSourcePoint(),
    fOrder(),
    fVertex(),
    fOutward(),
    fAlongSide(),
    fEdge(),
    a_n(),
    fIqS_Map(),
    fIqL_Map(),
    fSolidAngle(0.),
    fH(0.),
    fAccuracy(1.e-2)
{}

KTRWGFunctions::KTRWGFunctions(KTElectrode* eSource, double* pPosition, double* pSource) :
    fElectrode(eSource),
    fNVerticies(0),
    fQmax(16),
    fQmin(8),
    fPosition(pPosition),
    fCenter(),
    fDeltaR(),
    fNormal(),
    fRho(),
    fSourcePoint(pSource),
    fOrder(),
    fVertex(),
    fOutward(),
    fAlongSide(),
    fEdge(),
    a_n(),
    fIqS_Map(),
    fIqL_Map(),
    fSolidAngle(0.),
    fH(0.),
    fAccuracy(1.e-8)
{}

KTRWGFunctions::~KTRWGFunctions() {}

void KTRWGFunctions::SetElectrode(KTElectrode* eSource)
{
    fElectrode = eSource;
}

void KTRWGFunctions::SetPosition(double* pPosition)
{
    fPosition = pPosition;
}

void KTRWGFunctions::SetSourcePoint(double* pSource)
{
    fSourcePoint = pSource;
}

void KTRWGFunctions::Clear()
{
    fOrder.clear();
    fVertex.clear();
    fOutward.clear();
    fAlongSide.clear();
    fEdge.clear();
    a_n.clear();
    fIqS_Map.clear();
    fIqL_Map.clear();

    fDeltaR.Clear();
    fNormal.Clear();
    fRho.Clear();
    fCenter.Clear();
}

void KTRWGFunctions::InitVectors()
{

    Clear();

    //! Store center of electrode

    double P[3];
    fElectrode->Centroid(P, false);
    fCenter = P;
    fDeltaR = fPosition - fCenter;

    //!	Set up number of verticies in class

    fNVerticies = fElectrode->GetNVertices();

    //!	Set up normal vector to surface

    double n[3];
    for (unsigned int k = 0; k < 3; k++)
        n[k] = fElectrode->GetNormal(P, k);
    fNormal = n;

    //!  Solve for the height, h, orthogonal to the plan

    fH = fNormal.Dot(fDeltaR);

    //!  Set up vertex points to be used in calculations

    double tmp[3];
    std::vector<TVector3> tmpVector(fNVerticies);
    for (unsigned int i = 0; i < fNVerticies; i++) {
        fElectrode->GetVertex(i, tmp);
        tmpVector[i] = TVector3(tmp);
    }

    //! Check if normal and rotation match.  If not, switch orientation

    fOrder.resize(fNVerticies);
    for (unsigned int i = 0; i < fNVerticies; i++)
        fOrder.at(i) = i;

    if (fNVerticies == 3) {

        TVector3 b = ((tmpVector.at(1) - tmpVector.at(0)).Cross((tmpVector.at(2) - tmpVector.at(0)))).Unit();

        if (b.Dot(fNormal) < 0.) {
            fOrder.at(0) = 0;
            fOrder.at(1) = 2;
            fOrder.at(2) = 1;
        }
    }

    fVertex.reserve(fNVerticies);
    for (unsigned int i = 0; i < fNVerticies; i++)
        fVertex.push_back(tmpVector.at(fOrder.at(i)));

    //!	 Set up outward boundary vector

    for (unsigned int i = 0; i < fNVerticies; i++) {
        unsigned int index1 = i % fNVerticies;
        unsigned int index2 = (i + 1) % fNVerticies;
        fAlongSide.push_back((fVertex.at(index2) - fVertex.at(index1)));
        fEdge.push_back(0.5 * (fVertex.at(index2) + fVertex.at(index1)));
        fOutward.push_back((fNormal.Cross(fAlongSide.at(i))).Unit());
    }

    //!	Find vector rho

    fRho = fPosition - fH * fNormal;

    //! Set up vector of (p_i - r) where p_i is vertex of surface.

    for (unsigned int i = 0; i < fNVerticies; i++)
        a_n.push_back((fVertex.at(i) - fPosition).Unit());

    //! Finally, calculate solid angle

    CalcSolidAngle();
}

double KTRWGFunctions::FunctionIqS(int qIndex)
{

    double aValue = 0.;

    if (GetIqS(qIndex, aValue))
        return aValue;

    int qStart = (qIndex + 3) % 2 - 3;
    for (int q = qStart; q <= qIndex; q += 2) {
        if (q == -3) {
            if (fabs(fH) > 0.)
                aValue = GetSolidAngle() / fH;
            else if (fDeltaR.Mag() == 0.)
                aValue = GetSolidAngle();
        }
        else if (q > -2) {
            aValue *= q / (q + 2.) * pow(fH, 2.);

            for (unsigned int i = 0; i < fNVerticies; i++) {
                unsigned int j = (i + 1) % fNVerticies;
                double t_i = (fOutward.at(i)).Dot(fPosition - fEdge.at(i));
                aValue -= 1. / (q + 2.) * t_i * FunctionIqL(q, i, j);
            }
        }
    }

    SetIqS(qIndex, aValue);

    return aValue;
}

double KTRWGFunctions::FunctionIqL(int qIndex, unsigned int iStart, unsigned int iEnd)
{

    double aValue = 0.;

    if (GetIqL(qIndex, iStart, iEnd, aValue))
        return aValue;

    double fSPlus = (fVertex.at(iStart) - fPosition).Dot((fAlongSide.at(iStart)).Unit());
    double fRPlus = (fVertex.at(iStart) - fPosition).Mag();

    double fSMinus = (fVertex.at(iEnd) - fPosition).Dot((fAlongSide.at(iStart)).Unit());
    double fRMinus = (fVertex.at(iEnd) - fPosition).Mag();

    double fR0 = sqrt(pow(fRPlus, 2.) - pow(fSPlus, 2.));

    if (fR0 > 0.) {
        int qStart = (qIndex + 1) % 2 - 1;
        for (int q = qStart; q <= qIndex; q += 2) {
            if (q == -1) {
                aValue = log(fRPlus + fSPlus) - log(fRMinus + fSMinus);
            }
            else {
                aValue *= q / (q + 1.) * pow(fR0, 2.);
                aValue += 1. / (q + 1.) * (fSPlus * pow(fRPlus, q) - fSMinus * pow(fRMinus, q));
            }
        }
    }

    SetIqL(qIndex, iStart, iEnd, aValue);

    return aValue;
}

void KTRWGFunctions::CalcSolidAngle()
{

    if (fNVerticies == 3) {
        double x = 1.;

        for (unsigned int i = 0; i < fNVerticies - 1; i++) {
            for (unsigned int j = i + 1; j < fNVerticies; j++) {
                x += (a_n.at(i)).Dot(a_n.at(j));
            }
        }

        TVector3 tmp_a = (a_n.at(1)).Cross(a_n.at(2));
        TVector3 tmp_b = (a_n.at(0));
        double y = fabs(tmp_a.Dot(tmp_b));

        fSolidAngle = fabs(2. * TMath::ATan2(y, x));
    }
    else {

        //!	This uses the general Girald spherical excess theorem to determine the solid angle of a polygon
        //! About 0.1% of the time, this yields the wrong answer because of a factor of 2pi.
        //! Needs to be fixed and/or studied.

        double angle_sum = 0.;
        for (unsigned int i = 0; i < fNVerticies; i++) {
            unsigned int ip = (i + 1) % fNVerticies;
            unsigned int im = (i == 0) ? fNVerticies - 1 : i - 1;
            double dalpha = CalcEulerAngles(i, im) - CalcEulerAngles(i, ip);
            if ((dalpha) < 0.)
                dalpha += TMath::TwoPi();
            angle_sum += dalpha;
        }
        fSolidAngle = angle_sum - TMath::Pi() * (fNVerticies - 2);
        fSolidAngle = acos(cos(fSolidAngle));
    }

    if (fDeltaR.Mag() == 0.)
        fSolidAngle = TMath::TwoPi();

    if (fH < 0.)
        fSolidAngle *= -1.;
}

double KTRWGFunctions::CalcEulerAngles(unsigned int pIndex, unsigned int qIndex)
{
    double aValue = 0.;

    double pTheta = (a_n.at(pIndex)).Theta() - TMath::PiOver2();
    double qTheta = (a_n.at(qIndex)).Theta() - TMath::PiOver2();

    double pPhi = (a_n.at(pIndex)).Phi();
    double qPhi = (a_n.at(qIndex)).Phi();

    double x = sin(qPhi - pPhi) * cos(qTheta);
    double y = sin(qTheta) * cos(pTheta) - cos(qTheta) * sin(pTheta) * cos(qPhi - pPhi);

    aValue = TMath::ATan2(x, y);

    return aValue;
}

void KTRWGFunctions::SetIqS(int index, double aValue)
{

    fIqS_Map[index] = aValue;
}

bool KTRWGFunctions::GetIqS(int index, double& aValue) const
{

    bool isValid = false;
    aValue = 0.;

    isValid = (fIqS_Map.find(index) != fIqS_Map.end());
    if (isValid)
        aValue = fIqS_Map.find(index)->second;

    return isValid;
}

void KTRWGFunctions::SetIqL(int index, unsigned int iStart, unsigned int iEnd, double aValue)
{

    int key = 100 * (index) + 10 * iStart + iEnd;
    fIqL_Map[key] = aValue;
}

bool KTRWGFunctions::GetIqL(int index, unsigned int iStart, unsigned int iEnd, double& aValue) const
{

    bool isValid = false;
    aValue = 0.;

    int key = 100 * (index) + 10 * iStart + iEnd;

    isValid = (fIqL_Map.find(key) != fIqL_Map.end());
    if (isValid)
        aValue = fIqL_Map.find(key)->second;

    return isValid;
}

std::vector<std::complex<double>> KTRWGFunctions::CalcGreenFunction(const double k, const unsigned int iType)
{
    std::vector<std::complex<double>> aValue(3);
    std::vector<std::complex<double>> saveFirst(3);
    std::vector<std::complex<double>> aValue_lookback(3);
    std::complex<double> coefficient(0., 0.);
    std::complex<double> im(0., 1.);

    double kConstant = 1. / (4. * M_PI);
    double kPhase = k * fDeltaR.Mag();
    double diff = 0.;
    std::complex<double> fPrecisionTrack(0., 0.);
    TVector3 KqNterm;

    int qStart = -1;
    int qEnd = std::min(fQmin, fQmax);
    for (int q = qStart; q < qEnd; q++) {
        TVector3 UnitVector(1., 0., 0.);
        TVector3 ZeroVector(0., 0., 0.);
        for (int l = qStart; l <= q; l++) {
            TVector3 iqlterm;

            coefficient =
                pow(-1., q - l) * pow(im, q + 1) * pow(k, l + 1) / TMath::Factorial(l + 1) / TMath::Factorial(q - l);
            coefficient *= pow(kPhase, q - l);

            KqNterm.Clear();

            if (abs(coefficient) > 0.) {
                switch (iType) {
                    case kScalar:
                        KqNterm = UnitVector * FunctionIqS(l);
                        break;
                    case kVectorField:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l + 2, i, j) * (1. / (l + 2.));
                        }
                        KqNterm = (iqlterm + (fRho - fSourcePoint) * FunctionIqS(l));
                        break;
                    case kGradient:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l, i, j);
                        }
                        if (fabs(fH) > 1.e-13)
                            KqNterm = (iqlterm - fH * l * fNormal * FunctionIqS(l - 2));
                        else
                            KqNterm = iqlterm;  // - l * fNormal * FunctionIqS(l-2);
                        break;
                    case kCurl:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l, i, j);
                        }
                        if (fabs(fH) > 1.e-13)
                            KqNterm = (iqlterm - fH * l * fNormal * FunctionIqS(l - 2));
                        else
                            KqNterm = iqlterm;  // - l * fNormal * FunctionIqS(l-2);
                        KqNterm = -(fPosition - fSourcePoint).Cross(KqNterm);
                        break;

                    case kCurlScalar:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += UnitVector * (fRho - fSourcePoint).Dot((fAlongSide.at(i)).Unit()) *
                                       FunctionIqL(l, i, j);
                            iqlterm += UnitVector * (1. / (l + 2.)) *
                                       (pow((fVertex.at(i) - fPosition).Mag(), l + 2) -
                                        pow((fVertex.at(j) - fPosition).Mag(), l + 2));
                        }
                        KqNterm = iqlterm;
                        break;

                    case kCurlVector:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l + 2, i, j) * (1. / (l + 2.));
                        }
                        KqNterm = (iqlterm + (fRho - fSourcePoint) * FunctionIqS(l));
                        KqNterm = fNormal.Cross(KqNterm);
                        break;

                    case kCurlNormal:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l, i, j);
                        }
                        KqNterm = fNormal * l * (FunctionIqS(l) - fH * fH * FunctionIqS(l - 2));
                        KqNterm += fNormal * ((fPosition - fSourcePoint).Dot(iqlterm));
                        break;

                    case kCurlTangent:
                        for (unsigned int i = 0; i < fNVerticies; i++) {
                            unsigned int j = (i + 1) % fNVerticies;
                            iqlterm += fOutward.at(i) * FunctionIqL(l, i, j);
                        }
                        KqNterm = -fH * (iqlterm + l * (fRho - fSourcePoint) * FunctionIqS(l - 2));
                        break;
                    default:
                        KqNterm = ZeroVector;
                        break;
                }
                for (unsigned int i = 0; i < 3; i++)
                    aValue[i] += coefficient * KqNterm[i];
                if (q > -1)
                    fPrecisionTrack += coefficient * pow(fDeltaR.Mag(), l);
            }
        }
        diff = 0.;

        if (q < 1)
            for (unsigned int i = 0; i < 3; i++)
                saveFirst[i] = aValue[i];

        for (unsigned int i = 0; i < 3; i++) {
            if (abs(aValue_lookback[i]) > 0.)
                diff += abs(aValue_lookback[i] - aValue[i]) / abs(aValue_lookback[i]);
        }
        if (diff > fAccuracy)
            qEnd = TMath::Min(q + 2, fQmax);

        aValue_lookback = aValue;
    }

    if (((diff > fAccuracy) && (qEnd == fQmax)) && (abs(fPrecisionTrack) > fAccuracy)) {
        for (int i = 0; i < 3; i++)
            aValue[i] = saveFirst[i] * exp(im * kPhase);
    }

    for (int i = 0; i < 3; i++) {
        aValue[i] *= kConstant * exp(im * kPhase);
    }

    return aValue;
}

double KTRWGFunctions::Potential(double* pPosition, bool /*sym*/, double k)
{

    double aValue = 0.;
    std::vector<std::complex<double>> qPotential;

    bool isEqual = true;
    for (int i = 0; i < 3; i++)
        if (pPosition[i] != fPosition[i])
            isEqual = false;

    if (!isEqual) {
        SetPosition(pPosition);
        InitVectors();
    }

    qPotential = CalcGreenFunction(k, kScalar);
    aValue = real(qPotential[0] / KTElectrode::fEps0);

    return aValue;
}

void KTRWGFunctions::EField(double* pPosition, double* field, bool /*sym*/, double k)
{

    std::vector<std::complex<double>> qGradient;

    bool isEqual = true;
    for (int i = 0; i < 3; i++)
        if (pPosition[i] != fPosition[i])
            isEqual = false;

    if (!isEqual) {
        SetPosition(pPosition);
        InitVectors();
    }

    qGradient = CalcGreenFunction(k, kGradient);

    for (int i = 0; i < 3; i++)
        field[i] = real(qGradient[i] / KTElectrode::fEps0);
}

double KTRWGFunctions::GetAreaFromIntegral()
{

    return FunctionIqS(0);
}

std::vector<std::complex<double>> KTRWGFunctions::GetVectorIntegral()
{


    std::vector<std::complex<double>> qVectorIntegral(3);

    TVector3 iqlterm;
    for (unsigned int i = 0; i < fNVerticies; i++) {
        unsigned int j = (i + 1) % fNVerticies;
        iqlterm += 0.5 * fOutward.at(i) * FunctionIqL(2, i, j);
    }
    iqlterm += (fRho - fSourcePoint) * FunctionIqS(0);

    for (int k = 0; k < 3; k++)
        qVectorIntegral[k] = iqlterm[k];

    return qVectorIntegral;
}

std::vector<Complex_t> KTRWGFunctions::GetNumericalIntegral(const double k, const unsigned int iType, bool isSmooth)
{

    const unsigned int nDim = 3;
    std::vector<Complex_t> GreenIntegral(nDim);
    double theResult[2 * nDim];

    FieldNumerical theFieldSystem;
    theFieldSystem.SetOrigin(fCenter);
    theFieldSystem.SetElectrode(fElectrode);
    theFieldSystem.SetWaveNumber(k);
    theFieldSystem.SetSmooth(isSmooth);

    //create a new electrode integrator
    KVMElectrodeIntegrator* elecFieldInt = new KVMElectrodeIntegrator();

    //set the electrode, in this case we pass a triangle electrode
    elecFieldInt->SetElectrode(fElectrode);

    // set the relative tolerance
    elecFieldInt->SetRelativeTolerance(fAccuracy);

    typedef KVMFieldWrapper<FieldNumerical, &FieldNumerical::GetVectorField> functionVectorikR;
    functionVectorikR* theVectorFunction = new functionVectorikR(&theFieldSystem, 3, 6);

    typedef KVMFieldWrapper<FieldNumerical, &FieldNumerical::GetGradient> functionGradientikR;
    functionGradientikR* theGradientFunction = new functionGradientikR(&theFieldSystem, 3, 6);

    typedef KVMFieldWrapper<FieldNumerical, &FieldNumerical::GetCurl> functionCurlikR;
    functionCurlikR* theCurlFunction = new functionCurlikR(&theFieldSystem, 3, 6);

    switch (iType) {
        case kVectorField:
            elecFieldInt->SetField(theVectorFunction);
            break;
        case kGradient:
            elecFieldInt->SetField(theGradientFunction);
            break;
        case kCurl:
            elecFieldInt->SetField(theCurlFunction);
            break;
        default:
            return GreenIntegral;
            break;
    }

    //here we wrap a function which takes a point in 3-space and returns 6 values

    //set the (wrapped) function we want to integrate
    //perform the integration
    elecFieldInt->Integral(theResult);

    for (unsigned int j = 0; j < nDim; j++) {
        GreenIntegral[j].real() = theResult[2 * j];
        GreenIntegral[j].imag() = theResult[2 * j + 1];
    }

    return GreenIntegral;
}

std::vector<Complex_t> KTRWGFunctions::GetSmoothIntegral(const double k, const unsigned int iType)
{

    const unsigned int nDim = 3;
    std::vector<Complex_t> Gs(nDim), Ge(nDim), GreenIntegral(nDim);

    //!	Get Numerical value of smooth integral
    Gs = GetNumericalIntegral(k, iType, true);

    //! Get exact value of remaining function (i.e 1/R + ik - k^2*R/2)
    SetQMax(1);
    Ge = CalcGreenFunction(k, iType);

    //! Combine results
    for (unsigned int j = 0; j < nDim; j++)
        GreenIntegral[j] = Gs[j] + real(Ge[j]);

    return GreenIntegral;
}
