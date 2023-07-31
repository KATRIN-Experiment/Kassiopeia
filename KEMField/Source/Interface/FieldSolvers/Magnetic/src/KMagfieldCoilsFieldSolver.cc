/*
 * KMagfieldCoilsFieldSolver.cc
 *
 *  Created on: 31 Jan 2023
 *      Author: Jan Behrens
 */

#include "KMagfieldCoilsFieldSolver.hh"
#include "../../../Fields/Magnetic/include/KMagneticField.hh"

#include "KEMCoreMessage.hh"
#include "KTextFile.h"
#include "KRotation.hh"

namespace KEMField
{


KMagfieldCoilsFieldSolver::KMagfieldCoilsFieldSolver() :
    fNElliptic(32),
    fNMax(500),
    fEpsTol(1.e-8),
    fApproxExecCount(0),
    fDirectExecCount(0)
{}

std::string KMagfieldCoilsFieldSolver::WriteCoilFile(katrin::KTextFile* aFile, KElectromagnetContainer& aContainer)
{ 
    aFile->Open(katrin::KFile::eWrite);
    if (!aFile->IsOpen()) {
        kem_cout(eWarning) << "magfield coil file could not be opened" << eom;
        return "";
    }

   auto tFileName = aFile->GetName();

    //     First line: number of coils  (Ncoil).
    //     Then there are Ncoil number of lines; each line contains:
    //       cu  Cx Cy Cz  alpha  beta  tu  L Rmin Rmax
    //    cu: current of coil (A)
    //    Cx:  x component of coil center (m)
    //    Cy:  y component of coil center (m)
    //    Cz:  z component of coil center (m)
    //    alpha, beta:  coil direction Euler angles in degrees.
    //    Coil direction unit vector (ux, uy, uz) is defined by
    //           ux=sin(beta)*sin(alpha),   uy=-sin(beta)*cos(alpha),   uz=cos(beta),
    //  (e.g. beta=0: u in+z direction;  beta=180:  u in -z direction)
    //    tu: number of coil turns (double)
    //    L: coil length (m)
    //    Rmin:  inner radius of coil   (m)
    //    Rmax:  outer radius of coil  (m).

    kem_cout(eDebug) << "writing magfield coils to file: " << tFileName << eom;
    auto* tStream = aFile->File();

    unsigned tNumCoils = 0;
    for (size_t i = 0; i < aContainer.size(); ++i ) {
        auto tCoil = dynamic_cast<KCoil*>(aContainer.at(i));
        if (tCoil)
            tNumCoils++;
    }

    *tStream << tNumCoils << "\n";

    for (size_t i = 0; i < aContainer.size(); ++i ) {
        auto tCoil = dynamic_cast<KCoil*>(aContainer.at(i));
        if (tCoil) {

            // do not use coil->GetP0|P1() because it is defined as (r,0,z); z-axis is flipped here!
            auto p0 = tCoil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tCoil->GetZ0()));
            auto p1 = tCoil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tCoil->GetZ1()));

            katrin::KThreeVector tCenter = 0.5 * (p1 + p0);
            double tLength = (p1 - p0).Magnitude();

#if 1
            double tAlpha, tBeta, tGamma;
            katrin::KRotation tRotation;
            tRotation.SetRotatedFrame(tCoil->GetCoordinateSystem().GetXAxis(),
                                      tCoil->GetCoordinateSystem().GetYAxis(),
                                      tCoil->GetCoordinateSystem().GetZAxis());
            tRotation.GetEulerAnglesInDegrees(tAlpha, tBeta, tGamma);
#else
            // from Ferenc's CoilInputTransformMod.cc (coordinate conversion)
            {
                double ux = (p1.X() - p0.X()) / tLength;
                double uy = (p1.Y() - p0.Y()) / tLength;
                double uz = (p1.Z() - p0.Z()) / tLength;
                double ur = sqrt(ux*ux+uy*uy);

                double beta = 180./M_PI * acos(uz);
                double alpha;
                const double eps = 1.e-11;
                if (ur < eps) {
                    alpha = 0.;
                }
                else {
                    double uxur = ux/ur;
                    if (uxur > 1.)
                        uxur = 1.;
                    if (uxur < -1.)
                        uxur = -1.;
                     if (uy < eps)
                       alpha = 180./M_PI * asin(uxur);
                     else
                       alpha  = 180. - 180./M_PI * asin(uxur);
                }
#if 0  // DEBUGGING
                if (alpha < 0.)
                    alpha += 360.;
                if (tAlpha < 0.)
                    tAlpha += 360.;
                if (alpha != tAlpha || beta != tBeta) {
                    cout << "i, ux, uy, uz=       " << i << scientific << "      " << ux << "      " << uy<< "      " << uz << endl;
                    cout << "i, alpha, beta=       " << i << scientific << "      " << alpha << "      " << beta<< endl;
                    cout << "i, tAlpha, tBeta=       " << i << scientific << "      " << tAlpha << "      " << tBeta<< endl;
                    cout << endl;
#endif
                }
                tAlpha = alpha;
                tBeta = beta;
            }
#endif

            *tStream << std::scientific << std::setprecision(12)
                     << tCoil->GetCurrentPerTurn() << "\t"
                     << tCenter.X() << "\t"
                     << tCenter.Y() << "\t"
                     << -1*tCenter.Z() << "\t"
                     << tAlpha << "\t"
                     << tBeta << "\t"
                     << tCoil->GetNumberOfTurns() << "\t"
                     << tLength << "\t"
                     << tCoil->GetR0() << "\t"
                     << tCoil->GetR1() << "\t"
                     << "\n";
        }
    }

    aFile->Close();
    return tFileName;
}

void KMagfieldCoilsFieldSolver::InitializeCore(KElectromagnetContainer& aContainer)
{
    if (fSolver)
        return;

    string tDirName = fDirName.empty() ? SCRATCH_DEFAULT_DIR : fDirName;
    if (tDirName.at(tDirName.length()-1) != '/')
        tDirName += '/';

    string tObjectName = fObjectName.empty() ? GetFieldObject()->GetName() : fObjectName;
    tObjectName += '_';

    auto tCoilFile = katrin::KTextFile::CreateOutputTextFile(tDirName, fCoilFileName);
    string tCoilFileName = tDirName + fCoilFileName;

    if (fReplaceFile) {
        tCoilFileName = WriteCoilFile(tCoilFile, aContainer);  // update filename of created file
        kem_cout(eNormal) << "initializing MagfieldCoils solver with file <" << tCoilFileName << ">" << eom;
    }
    else {
        if (katrin::KFile::Test(tCoilFileName))
            kem_cout(eNormal) << "initializing MagfieldCoils solver with existing file <" << tCoilFileName << ">" << eom;
        else
            kem_cout(eError) << "cannot initialize MagfieldCoils solver with missing file <" << tCoilFileName << ">" << eom;
    }

    kem_cout_debug("sourcepoint files are begin written to <" << tDirName + tObjectName+"*.txt" << eom);
    fSolver = std::make_shared<MagfieldCoils>(tDirName, tObjectName, tCoilFileName, fNElliptic, fNMax, fEpsTol);
}

void KMagfieldCoilsFieldSolver::DeinitializeCore()
{
    const auto& A = fApproxExecCount;
    const auto& D = fDirectExecCount;
    const auto& T = fApproxExecCount+fDirectExecCount;
    kem_cout(eNormal) << "MagfieldCoils solver execution counts:" << ret
                      << "approx:  " << A << " (" << std::floor(100.*A/T) << "%)" << ret
                      << "direct:  " << D << " (" << std::floor(100.*D/T) << "%)" << eom;
}

KFieldVector KMagfieldCoilsFieldSolver::MagneticPotentialCore(const KPosition& /*P*/) const
{
    return KFieldVector::sInvalid;
}

KFieldVector KMagfieldCoilsFieldSolver::MagneticFieldCore(const KPosition& P) const
{
    KFieldVector b;
    if (fForceElliptic) {
        fSolver->MagfieldElliptic(P.Components(), b.Components());
        fDirectExecCount++;
        return b;
    }

    if (! fSolver->Magfield(P.Components(), b.Components())) {  // automatic fallback to elliptic
        fDirectExecCount++;
        kem_cout_debug("MagfieldCoils solver falling back to direct integration at point <"
                        << P.Z() << " " << P.Perp() << ">" << eom);
    }
    else {
        fApproxExecCount++;
    }
    return b;
}

KGradient KMagfieldCoilsFieldSolver::MagneticGradientCore(const KPosition& P) const
{
    KGradient g;
    double epsilon = 1.e-6;
    for (unsigned int i = 0; i < 3; i++) {
        KPosition Pplus = P;
        Pplus[i] += epsilon;
        KPosition Pminus = P;
        Pminus[i] -= epsilon;
        KFieldVector Bplus = MagneticField(Pplus);
        KFieldVector Bminus = MagneticField(Pminus);
        for (unsigned int j = 0; j < 3; j++)
            g[j + 3 * i] = (Bplus[j] - Bminus[j]) / (2. * epsilon);
    }
    return g;
}

} /* namespace KEMField */
