#include "KGElectromagnetConverter.hh"

#include "KEMCoreMessage.hh"
using KEMField::kem_cout;

#include "KCoilIntegrator.hh"
#include "KGCylinderSpace.hh"
#include "KGCylinderSurface.hh"
#include "KGRodSpace.hh"

using katrin::KThreeMatrix;
using katrin::KThreeVector;

namespace KGeoBag
{

KGElectromagnetConverter::KGElectromagnetConverter() :
    fElectromagnetContainer(nullptr),
    fMagfield3File(nullptr),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit),
    fCurrentElectromagnetSpace(nullptr),
    fCurrentElectromagnetSurface(nullptr)
{}
KGElectromagnetConverter::~KGElectromagnetConverter()
{
    if (fMagfield3File && fMagfield3File->IsOpen())
        fMagfield3File->Close();
}

void KGElectromagnetConverter::SetDumpMagfield3ToFile(const std::string& aDirectory, const std::string& aFileName)
{
    fMagfield3File = katrin::KTextFile::CreateOutputTextFile(aDirectory, aFileName);
    if (!fMagfield3File)
        return;

    fMagfield3File->Open(katrin::KFile::eWrite);
    if (!fMagfield3File->IsOpen()) {
        kem_cout(eWarning) << "magfield3 file could not be opened" << eom;
        return;
    }

    kem_cout() << "Saving magfield3 geometry to file: " << fMagfield3File->GetName() << eom;

    auto* tStream = fMagfield3File->File();
    (*tStream) << '#' << "cur_dens" << '\t' << "x0" << '\t' << "y0" << '\t' << "z0" << '\t' << "x1" << '\t' << "y1"
               << '\t' << "z1" << '\t' << "r0" << '\t' << "r1" << '\t' << "int_scale" << '\t' << "current" << '\t'
               << "num_turns"
               << "\t"
               << "name" << std::endl;
}

void KGElectromagnetConverter::SetSystem(const KThreeVector& anOrigin, const KThreeVector& anXAxis,
                                         const KThreeVector& aYAxis, const KThreeVector& aZAxis)
{
    fOrigin = anOrigin;
    fXAxis = anXAxis;
    fYAxis = aYAxis;
    fZAxis = aZAxis;
    return;
}
const KThreeVector& KGElectromagnetConverter::GetOrigin() const
{
    return fOrigin;
}
const KThreeVector& KGElectromagnetConverter::GetXAxis() const
{
    return fXAxis;
}
const KThreeVector& KGElectromagnetConverter::GetYAxis() const
{
    return fYAxis;
}
const KThreeVector& KGElectromagnetConverter::GetZAxis() const
{
    return fZAxis;
}

KThreeVector KGElectromagnetConverter::GlobalToInternalPosition(const KThreeVector& aVector)
{
    KThreeVector tPosition(aVector - fOrigin);
    return KThreeVector(tPosition.Dot(fXAxis), tPosition.Dot(fYAxis), tPosition.Dot(fZAxis));
}
KThreeVector KGElectromagnetConverter::GlobalToInternalVector(const KThreeVector& aVector)
{
    const KThreeVector& tVector(aVector);
    return KThreeVector(tVector.Dot(fXAxis), tVector.Dot(fYAxis), tVector.Dot(fZAxis));
}
KThreeVector KGElectromagnetConverter::InternalToGlobalPosition(const KThreeVector& aVector)
{
    KThreeVector tPosition(aVector.X(), aVector.Y(), aVector.Z());
    return KThreeVector(fOrigin + tPosition.X() * fXAxis + tPosition.Y() * fYAxis + tPosition.Z() * fZAxis);
}
KThreeVector KGElectromagnetConverter::InternalToGlobalVector(const KThreeVector& aVector)
{
    KThreeVector tVector(aVector.X(), aVector.Y(), aVector.Z());
    return KThreeVector(tVector.X() * fXAxis + tVector.Y() * fYAxis + tVector.Z() * fZAxis);
}

KThreeMatrix KGElectromagnetConverter::InternalTensorToGlobal(const KEMField::KGradient& aGradient)
{
    KThreeMatrix
        tTransform(fXAxis[0], fYAxis[0], fZAxis[0], fXAxis[1], fYAxis[1], fZAxis[1], fXAxis[2], fYAxis[2], fZAxis[2]);

    tTransform = tTransform.Multiply(aGradient.MultiplyTranspose(tTransform));
    KThreeMatrix tThreeMatrix(tTransform[0],
                              tTransform[1],
                              tTransform[2],
                              tTransform[3],
                              tTransform[4],
                              tTransform[5],
                              tTransform[6],
                              tTransform[7],
                              tTransform[8]);
    return tThreeMatrix;
}

void KGElectromagnetConverter::VisitSpace(KGSpace* aSpace)
{
    Clear();

#ifdef PRINT_MAGFIELD3
    //std::cout << "# " << aSpace->GetPath() << std::endl;
#endif

    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();

    return;
}

void KGElectromagnetConverter::VisitSurface(KGSurface* aSurface)
{
    Clear();

    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();

    return;
}

void KGElectromagnetConverter::VisitExtendedSpace(KGExtendedSpace<KGElectromagnet>* electromagnetSpace)
{
    fCurrentElectromagnetSpace = electromagnetSpace;
}

void KGElectromagnetConverter::VisitExtendedSurface(KGExtendedSurface<KGElectromagnet>* electromagnetSurface)
{
    fCurrentElectromagnetSurface = electromagnetSurface;
}

void KGElectromagnetConverter::VisitWrappedSpace(KGRodSpace* rod)
{
    if (fCurrentElectromagnetSpace) {
        double tCurrent = fCurrentElectromagnetSpace->GetCurrent();

        if (fabs(tCurrent) < 1e-12)
            kem_cout(eInfo) << "adding line current with no current defined: " << fCurrentElectromagnetSpace->GetName()
                            << eom;

        for (unsigned int i = 0; i < rod->GetObject()->GetNCoordinates() - 1; i++) {
            KEMField::KPosition p0(rod->GetObject()->GetCoordinate(i, 0),
                         rod->GetObject()->GetCoordinate(i, 1),
                         rod->GetObject()->GetCoordinate(i, 2));
            KEMField::KPosition p1(rod->GetObject()->GetCoordinate(i + 1, 0),
                         rod->GetObject()->GetCoordinate(i + 1, 1),
                         rod->GetObject()->GetCoordinate(i + 1, 2));

            auto* lineCurrent = new KEMField::KLineCurrent();
            lineCurrent->SetValues(p0, p1, tCurrent);

            lineCurrent->GetCoordinateSystem().SetValues(GlobalToInternalPosition(fCurrentOrigin),
                                                         GlobalToInternalVector(fCurrentXAxis),
                                                         GlobalToInternalVector(fCurrentYAxis),
                                                         GlobalToInternalVector(fCurrentZAxis));
            fElectromagnetContainer->push_back(lineCurrent);
        }
    }
}

void KGElectromagnetConverter::VisitCylinderSurface(KGCylinderSurface* cylinder)
{
    if (fCurrentElectromagnetSurface) {
        double tR = cylinder->R();
        double tZMin = cylinder->Z1() > cylinder->Z2() ? cylinder->Z2() : cylinder->Z1();
        double tZMax = cylinder->Z1() > cylinder->Z2() ? cylinder->Z1() : cylinder->Z2();
        double tCurrent = fCurrentElectromagnetSurface->GetCurrent();
        //double tNumTurns = fCurrentElectromagnetSurface->GetCurrentTurns();

        if (fabs(tCurrent) < 1e-12)
            kem_cout(eInfo) << "adding solenoid with no current defined: " << fCurrentElectromagnetSurface->GetName()
                            << eom;

        auto* solenoid = new KEMField::KSolenoid();
        solenoid->SetValues(tR, tZMin, tZMax, tCurrent);

        solenoid->GetCoordinateSystem().SetValues(GlobalToInternalPosition(fCurrentOrigin),
                                                  GlobalToInternalVector(fCurrentXAxis),
                                                  GlobalToInternalVector(fCurrentYAxis),
                                                  GlobalToInternalVector(fCurrentZAxis));
        fElectromagnetContainer->push_back(solenoid);
    }
}

void KGElectromagnetConverter::VisitCylinderTubeSpace(KGCylinderTubeSpace* cylinderTube)
{
    if (fCurrentElectromagnetSpace) {
        int tNDisc = (int) cylinderTube->RadialMeshCount();
        double tRMin = cylinderTube->R1() > cylinderTube->R2() ? cylinderTube->R2() : cylinderTube->R1();
        double tRMax = cylinderTube->R1() > cylinderTube->R2() ? cylinderTube->R1() : cylinderTube->R2();
        double tZMin = cylinderTube->Z1() > cylinderTube->Z2() ? cylinderTube->Z2() : cylinderTube->Z1();
        double tZMax = cylinderTube->Z1() > cylinderTube->Z2() ? cylinderTube->Z1() : cylinderTube->Z2();
        double tCurrent = fCurrentElectromagnetSpace->GetCurrent();

        if (fabs(tCurrent) < 1e-12)
            kem_cout(eInfo) << "adding coil with no current defined: " << fCurrentElectromagnetSpace->GetName() << eom;

        auto* coil = new KEMField::KCoil();
        coil->SetValues(tRMin, tRMax, tZMin, tZMax, tCurrent, tNDisc);

        coil->GetCoordinateSystem().SetValues(GlobalToInternalPosition(fCurrentOrigin),
                                              GlobalToInternalVector(fCurrentXAxis),
                                              GlobalToInternalVector(fCurrentYAxis),
                                              GlobalToInternalVector(fCurrentZAxis));
        fElectromagnetContainer->push_back(coil);

        if (fMagfield3File && fMagfield3File->IsOpen()) {
            auto* tStream = fMagfield3File->File();

            // do not use coil->GetP0|P1() because it is defined as (r,0,z)
            auto p0 = coil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tZMin));
            auto p1 = coil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tZMax));

            double tLineCurrent = fCurrentElectromagnetSpace->GetLineCurrent();
            double tNumTurns = fCurrentElectromagnetSpace->GetCurrentTurns();
            std::string tName = fCurrentElectromagnetSpace->GetName();

            (*tStream) << ' ' << coil->GetCurrentDensity() << '\t' << p0.X() << '\t' << p0.Y() << '\t' << p0.Z() << '\t'
                       << p1.X() << '\t' << p1.Y() << '\t' << p1.Z() << '\t' << coil->GetR0() << '\t' << coil->GetR1()
                       << '\t' << coil->GetIntegrationScale() << '\t' << tLineCurrent << '\t' << tNumTurns << "\t# "
                       << tName << std::endl;
        }
    }
}

void KGElectromagnetConverter::Clear()
{
    fCurrentElectromagnetSpace = nullptr;
    fCurrentElectromagnetSurface = nullptr;
    return;
}

}  // namespace KGeoBag
