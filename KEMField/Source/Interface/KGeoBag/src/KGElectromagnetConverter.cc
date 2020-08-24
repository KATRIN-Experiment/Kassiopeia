#include "KGElectromagnetConverter.hh"

#include "KCoilIntegrator.hh"
#include "KGCylinderSpace.hh"
#include "KGCylinderSurface.hh"
#include "KGRodSpace.hh"

/// FIXME: If defined, print magfield3 lines of converted magnet geometry. This should be a runtime option.
//#define PRINT_MAGFIELD3

namespace KGeoBag
{

KGElectromagnetConverter::KGElectromagnetConverter() :
    fElectromagnetContainer(nullptr),
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
KGElectromagnetConverter::~KGElectromagnetConverter() {}

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
    KThreeVector tVector(aVector);
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

KThreeMatrix KGElectromagnetConverter::InternalTensorToGlobal(const KGradient& aGradient)
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
#ifdef PRINT_MAGFIELD3
    //std::cout << "# " << electromagnetSpace->GetName() << std::endl;
#endif

    fCurrentElectromagnetSpace = electromagnetSpace;
}

void KGElectromagnetConverter::VisitExtendedSurface(KGExtendedSurface<KGElectromagnet>* electromagnetSurface)
{
    fCurrentElectromagnetSurface = electromagnetSurface;
}

void KGElectromagnetConverter::VisitWrappedSpace(KGRodSpace* rod)
{
    if (fCurrentElectromagnetSpace) {
        for (unsigned int i = 0; i < rod->GetObject()->GetNCoordinates() - 1; i++) {
            KPosition p0(rod->GetObject()->GetCoordinate(i, 0),
                         rod->GetObject()->GetCoordinate(i, 1),
                         rod->GetObject()->GetCoordinate(i, 2));
            KPosition p1(rod->GetObject()->GetCoordinate(i + 1, 0),
                         rod->GetObject()->GetCoordinate(i + 1, 1),
                         rod->GetObject()->GetCoordinate(i + 1, 2));

            auto* lineCurrent = new KLineCurrent();
            lineCurrent->SetValues(p0, p1, fCurrentElectromagnetSpace->GetCurrent());

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

        auto* solenoid = new KSolenoid();
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
        unsigned int tNDisc = cylinderTube->RadialMeshCount();
        double tRMin = cylinderTube->R1() > cylinderTube->R2() ? cylinderTube->R2() : cylinderTube->R1();
        double tRMax = cylinderTube->R1() > cylinderTube->R2() ? cylinderTube->R1() : cylinderTube->R2();
        double tZMin = cylinderTube->Z1() > cylinderTube->Z2() ? cylinderTube->Z2() : cylinderTube->Z1();
        double tZMax = cylinderTube->Z1() > cylinderTube->Z2() ? cylinderTube->Z1() : cylinderTube->Z2();
        double tCurrent = fCurrentElectromagnetSpace->GetCurrent();

        auto* coil = new KCoil();
        coil->SetValues(tRMin, tRMax, tZMin, tZMax, tCurrent, tNDisc);

        coil->GetCoordinateSystem().SetValues(GlobalToInternalPosition(fCurrentOrigin),
                                              GlobalToInternalVector(fCurrentXAxis),
                                              GlobalToInternalVector(fCurrentYAxis),
                                              GlobalToInternalVector(fCurrentZAxis));
        fElectromagnetContainer->push_back(coil);

#ifdef PRINT_MAGFIELD3
        // do not use coil->GetP0|P1() because it is defined as (r,0,z)
        auto p0 = coil->GetCoordinateSystem().ToGlobal(KPosition(0, 0, tZMin));
        auto p1 = coil->GetCoordinateSystem().ToGlobal(KPosition(0, 0, tZMax));

        double tLineCurrent = fCurrentElectromagnetSpace->GetLineCurrent();
        double tNumTurns = fCurrentElectromagnetSpace->GetCurrentTurns();
        std::string tName = fCurrentElectromagnetSpace->GetName();

        std::cout << " " << coil->GetCurrentDensity() << " " << p0.X() << " " << p0.Y() << " " << p0.Z() << " "
                  << p1.X() << " " << p1.Y() << " " << p1.Z() << " " << coil->GetR0() << " " << coil->GetR1() << " "
                  << coil->GetIntegrationScale() << " " << tLineCurrent << " " << tNumTurns << "\t# " << tName << std::endl;
#endif
    }
}

void KGElectromagnetConverter::Clear()
{
    fCurrentElectromagnetSpace = nullptr;
    fCurrentElectromagnetSurface = nullptr;
    return;
}

}  // namespace KGeoBag
