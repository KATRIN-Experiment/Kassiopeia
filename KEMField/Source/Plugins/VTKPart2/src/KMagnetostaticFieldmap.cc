/*
 * KMagnetostaticFieldmap.cc
 *
 *  Created on: 26 Apr 2016
 *      Author: wolfgang
 */

#include "KMagnetostaticFieldmap.hh"

#include "KEMCout.hh"
#include "KEMFileInterface.hh"
#include "KEMSimpleException.hh"
#include "KFile.h"
#include "KGslErrorHandler.h"

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>

using std::string;

namespace KEMField
{


KMagfieldMapVTK::KMagfieldMapVTK(const string& aFilename)
{
    cout << "loading field map from file <" << aFilename << ">" << endl;

    vtkXMLImageDataReader* reader = vtkXMLImageDataReader::New();
    reader->SetFileName(aFilename.c_str());
    reader->Update();
    fImageData = reader->GetOutput();

    int dims[3];
    double bounds[6];
    fImageData->GetDimensions(dims);
    fImageData->GetBounds(bounds);
    //fieldmsg_debug( "field map has " << fImageData->GetNumberOfPoints() << " points (" << dims[0] << "x" << dims[1] << "x" << dims[2] << ") and ranges from " << KFieldVector(bounds[0],bounds[2],bounds[4]) << " to " << KFieldVector(bounds[1],bounds[3],bounds[4]) << eom);
}

KMagfieldMapVTK::~KMagfieldMapVTK() = default;

bool KMagfieldMapVTK::GetValue(const string& array, const KPosition& aSamplePoint, double* aValue, bool gradient) const
{
    vtkDataArray* data = fImageData->GetPointData()->GetArray(array.c_str());
    if (data == nullptr) return false;

    // get coordinates of closest mesh point
    vtkIdType center = fImageData->FindPoint((double*) (aSamplePoint.Components()));
    if (center < 0)
        return false;

    // get value at center
    if (!gradient)
        data->GetTuple(center, aValue);

    return true;
}

bool KMagfieldMapVTK::GetField(const KPosition& aSamplePoint, const double& /*aSampleTime*/, KFieldVector& aField) const
{
    //fieldmsg_debug( "sampling magnetic field at point " << aSamplePoint << eom);

    double value[3];
    if (GetValue("magnetic field", aSamplePoint, value)) {
        aField.SetComponents(value);
        return true;
    }
    return false;
}

bool KMagfieldMapVTK::GetGradient(const KPosition& aSamplePoint, const double& /*aSampleTime*/,
                                  KGradient& aGradient, bool grad_numerical) const
{
    //fieldmsg_debug( "sampling magnetic gradient at point " << aSamplePoint << eom);

    double value[9];
    if (GetValue("magnetic gradient", aSamplePoint, value)) {
        aGradient.SetComponents(value);
        return true;
    }
    if (GetValue("magnetic field", aSamplePoint, value, true)) {
        aGradient.SetComponents(value);
        return true;
    }
    return false;
}

KLinearInterpolationMagfieldMapVTK::KLinearInterpolationMagfieldMapVTK(const string& aFilename) :
    KMagfieldMapVTK(aFilename)
{}

KLinearInterpolationMagfieldMapVTK::~KLinearInterpolationMagfieldMapVTK() = default;

bool KLinearInterpolationMagfieldMapVTK::GetValue(const string& array, const KPosition& aSamplePoint,
                                                  double* aValue, bool gradient) const
{
    vtkDataArray* data = fImageData->GetPointData()->GetArray(array.c_str());
    if (data == nullptr) return false;

    // get coordinates of surrounding mesh points
    static const char map[8][3] = {
        {0, 0, 0},  // c000
        {1, 0, 0},  // c100
        {0, 1, 0},  // c010
        {1, 1, 0},  // c110
        {0, 0, 1},  // c001
        {1, 0, 1},  // c101
        {0, 1, 1},  // c011
        {1, 1, 1},  // c111
    };
    static KFieldVector vertices[8];
    static double values
        [9]
        [8];  // always allocate for matrices even if we have scalars (to be safe) - note that array ordering is swapped

    double* spacing = fImageData->GetSpacing();
    //compute corner point of mesh cell aSamplePoint belongs to
    KFieldVector start_point = KFieldVector(floor(aSamplePoint.X() / spacing[0]) * spacing[0],
                                            floor(aSamplePoint.Y() / spacing[1]) * spacing[1],
                                            floor(aSamplePoint.Z() / spacing[2]) * spacing[2]);
    for (int i = 0; i < 8; i++) {
        // first compute the coordinates of the surrounding mesh points ...
        KFieldVector point =
            start_point + KFieldVector(map[i][0] * spacing[0], map[i][1] * spacing[1], map[i][2] * spacing[2]);
        vtkIdType corner = fImageData->FindPoint((double*) (point.Components()));
        if (corner < 0)
            return false;
        // ... then retrieve data at these points
        vertices[i] = fImageData->GetPoint(corner);
        for (int k = 0; k < data->GetNumberOfComponents(); k++)
            values[k][i] = data->GetComponent(corner, k);
    }

    // get interpolated value at center
    double xd = (aSamplePoint.X() - vertices[0][0]) / spacing[0];
    double yd = (aSamplePoint.Y() - vertices[0][1]) / spacing[1];
    double zd = (aSamplePoint.Z() - vertices[0][2]) / spacing[2];
    if (! gradient) {
        static int d[3] = {0, 0, 0};
        for (int k = 0; k < data->GetNumberOfComponents(); k++) {
            aValue[k] = _trilinearInterpolate(&(values[k][0]), d, xd, yd, zd);
        }
    } else {
        for (int l = 0; l < data->GetNumberOfComponents(); l++) {
            int d[3] = {0, 0, 0};
            d[l] = 1; // 1st derivative in component l
            for (int k = 0; k < data->GetNumberOfComponents(); k++) {
                aValue[3*l+k] = _trilinearInterpolate(&(values[k][0]), d, xd, yd, zd) / spacing[l];
            }
        }
    }

    return true;
}

double KLinearInterpolationMagfieldMapVTK::_linearInterpolate(double p[], int d[], double x)  // array of 2
{
    if (d[0] == 1)
        return p[1] - p[0];
    else
        return p[0] + x * (p[1] - p[0]);
}

double KLinearInterpolationMagfieldMapVTK::_bilinearInterpolate(double p[], int d[], double x, double y)  // array of 2x2
{
    static double q[2];
    q[0] = _linearInterpolate(&(p[0]), &(d[0]), x);
    q[1] = _linearInterpolate(&(p[2]), &(d[0]), x);
    return _linearInterpolate(q, &(d[1]), y);
}

double KLinearInterpolationMagfieldMapVTK::_trilinearInterpolate(double p[], int d[], double x, double y,
                                                                 double z)  // array of 2x2x2
{
    static double q[2];
    q[0] = _bilinearInterpolate(&(p[0]), &(d[0]), x, y);
    q[1] = _bilinearInterpolate(&(p[4]), &(d[0]), x, y);
    return _linearInterpolate(q, &(d[2]), z);
}
////////////////////////////////////////////////////////////////////

KCubicInterpolationMagfieldMapVTK::KCubicInterpolationMagfieldMapVTK(const string& aFilename) :
    KMagfieldMapVTK(aFilename)
{}

KCubicInterpolationMagfieldMapVTK::~KCubicInterpolationMagfieldMapVTK() = default;

bool KCubicInterpolationMagfieldMapVTK::GetValue(const string& array, const KPosition& aSamplePoint,
                                                 double* aValue, bool /* gradient */) const
{
    vtkDataArray* data = fImageData->GetPointData()->GetArray(array.c_str());
    if (data == nullptr) return false;

    // get coordinates of surrounding mesh points
    static const char map[64][3] = {
        //
        {-1, -1, -1},  // 00
        {-1, -1, 0},
        {-1, -1, 1},
        {-1, -1, 2},

        {-1, 0, -1},  // 04
        {-1, 0, 0},
        {-1, 0, 1},
        {-1, 0, 2},

        {-1, 1, -1},  // 08
        {-1, 1, 0},
        {-1, 1, 1},
        {-1, 1, 2},

        {-1, 2, -1},  // 12
        {-1, 2, 0},
        {-1, 2, 1},
        {-1, 2, 2},
        //
        {0, -1, -1},  // 16
        {0, -1, 0},
        {0, -1, 1},
        {0, -1, 2},

        {0, 0, -1},  // 20
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 2},

        {0, 1, -1},  // 24
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 2},

        {0, 2, -1},  // 28
        {0, 2, 0},
        {0, 2, 1},
        {0, 2, 2},
        //
        {1, -1, -1},  // 22
        {1, -1, 0},
        {1, -1, 1},
        {1, -1, 2},

        {1, 0, -1},  // 26
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 2},

        {1, 1, -1},  // 40
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 2},

        {1, 2, -1},  // 44
        {1, 2, 0},
        {1, 2, 1},
        {1, 2, 2},
        //
        {2, -1, -1},  // 48
        {2, -1, 0},
        {2, -1, 1},
        {2, -1, 2},

        {2, 0, -1},  // 52
        {2, 0, 0},
        {2, 0, 1},
        {2, 0, 2},

        {2, 1, -1},  // 56
        {2, 1, 0},
        {2, 1, 1},
        {2, 1, 2},

        {2, 2, -1},  // 60
        {2, 2, 0},
        {2, 2, 1},
        {2, 2, 2},
    };
    static KFieldVector vertices[64];
    static double values
        [9]
        [64];  // always allocate for matrices even if we have scalars (to be safe) - note that array ordering is swapped

    double* spacing = fImageData->GetSpacing();
    //compute corner point of mesh cell aSamplePoint belongs to
    KFieldVector start_point = KFieldVector(floor(aSamplePoint.X() / spacing[0]) * spacing[0],
                                            floor(aSamplePoint.Y() / spacing[1]) * spacing[1],
                                            floor(aSamplePoint.Z() / spacing[2]) * spacing[2]);
    for (int i = 0; i < 64; i++) {
        // first compute the coordinates of the surrounding mesh points ...
        KFieldVector point =
            start_point + KFieldVector(map[i][0] * spacing[0], map[i][1] * spacing[1], map[i][2] * spacing[2]);
        vtkIdType corner = fImageData->FindPoint((double*) (point.Components()));
        if (corner < 0)
            return false;
        // ... then retrieve data at these points
        vertices[i] = fImageData->GetPoint(corner);
        for (int k = 0; k < data->GetNumberOfComponents(); k++)
            values[k][i] = data->GetComponent(corner, k);
    }

    double xd =
        (aSamplePoint.X() - vertices[21][0]) / spacing[0];  // point index 21 is at -1/-1/-1 coords = "lower" corner
    double yd = (aSamplePoint.Y() - vertices[21][1]) / spacing[1];
    double zd = (aSamplePoint.Z() - vertices[21][2]) / spacing[2];
    static int d[3] = {0, 0, 0};
    for (int k = 0; k < data->GetNumberOfComponents(); k++) {
        aValue[k] = _tricubicInterpolate(&(values[k][0]), d, xd, yd, zd);
    }

    return true;
}

double KCubicInterpolationMagfieldMapVTK::_cubicInterpolate(double p[], int d[],
                                                            double x)  // array of 4
{
    if (d[0] == 1)
        return 0.5 *
                  (p[2] - p[0]) + 2. * x *
                       (2. * p[0] - 5. * p[1] + 4. * p[2] - p[3] + 3. * x *
                           (3. * (p[1] - p[2]) + p[3] - p[0]));
    else
        return p[1] +
               0.5 * x *
                   (p[2] - p[0] + x *
                       (2. * p[0] - 5. * p[1] + 4. * p[2] - p[3] + x *
                           (3. * (p[1] - p[2]) + p[3] - p[0])));
}

double KCubicInterpolationMagfieldMapVTK::_bicubicInterpolate(double p[], int d[],
                                                              double x, double y)  // array of 4x4
{
    static double q[4];
    q[0] = _cubicInterpolate(&(p[0]),  &(d[1]), y);
    q[1] = _cubicInterpolate(&(p[4]),  &(d[1]), y);
    q[2] = _cubicInterpolate(&(p[8]),  &(d[1]), y);
    q[3] = _cubicInterpolate(&(p[12]), &(d[1]), y);
    return _cubicInterpolate(q, &(d[0]), x);
}

double KCubicInterpolationMagfieldMapVTK::_tricubicInterpolate(double p[], int d[],
                                                               double x, double y, double z)  // array of 4x4x4
{
    static double q[4];
    q[0] = _bicubicInterpolate(&(p[0]),  &(d[1]), y, z);
    q[1] = _bicubicInterpolate(&(p[16]), &(d[1]), y, z);
    q[2] = _bicubicInterpolate(&(p[32]), &(d[1]), y, z);
    q[3] = _bicubicInterpolate(&(p[48]), &(d[1]), y, z);
    return _cubicInterpolate(q, &(d[0]), x);
}
////////////////////////////////////////////////////////////////////

KMagnetostaticFieldmap::KMagnetostaticFieldmap() :
    fDirectory(SCRATCH_DEFAULT_DIR),
    fInterpolation(0),
    fGradNumerical(false),
    fFieldMap(nullptr)
{}

KMagnetostaticFieldmap::~KMagnetostaticFieldmap() = default;

KFieldVector KMagnetostaticFieldmap::MagneticPotentialCore(const KPosition& /*P*/) const
{
    KFieldVector tPotential;
    tPotential.SetComponents(0., 0., 0.);
    return tPotential;
}

KFieldVector KMagnetostaticFieldmap::MagneticFieldCore(const KPosition& P) const
{
    KFieldVector tField;
    tField.SetComponents(0., 0., 0.);
    double aRandomTime = 0;
    if (!fFieldMap->GetField(P, aRandomTime, tField))
        cout << "WARNING: could not compute magnetic field at sample point " << P << endl;

    return tField;
}

KGradient KMagnetostaticFieldmap::MagneticGradientCore(const KPosition& P) const
{
    KGradient tGradient(KGradient::sZero);
    double aRandomTime = 0;
    if (!fFieldMap->GetGradient(P, aRandomTime, tGradient, fGradNumerical))
        cout << "WARNING: could not compute magnetic gradient at sample point " << P << endl;

    return tGradient;
}

void KMagnetostaticFieldmap::SetDirectory(const std::string& aDirectory)
{
    fDirectory = aDirectory;
    return;
}

void KMagnetostaticFieldmap::SetFile(const std::string& aFile)
{
    fFile = aFile;
    return;
}

void KMagnetostaticFieldmap::SetInterpolation(const string& aMode)
{
    if (aMode == string("none") || aMode == string("nearest"))
        fInterpolation = 0;
    else if (aMode == string("linear"))
        fInterpolation = 1;
    else if (aMode == string("cubic"))
        fInterpolation = 3;
    else
        fInterpolation = -1;
    return;
}

void KMagnetostaticFieldmap::SetGradNumerical(bool aFlag)
{
    fGradNumerical = aFlag;
}

void KMagnetostaticFieldmap::InitializeCore()
{
    string filename = fDirectory + "/" + fFile;

    /// one could use different data back-ends here (e.g. ROOT instead of VTK, or ASCII files ...)
    switch (fInterpolation) {
        case 0:
            fFieldMap = std::make_shared<KMagfieldMapVTK>(filename);
            break;
        case 1:
            fFieldMap = std::make_shared<KLinearInterpolationMagfieldMapVTK>(filename);
            break;
        case 3:
            fFieldMap = std::make_shared<KCubicInterpolationMagfieldMapVTK>(filename);
            break;
        default:
            throw KEMSimpleException("interpolation mode " + std::to_string(fInterpolation) + " is not implemented");
            break;
    }

    cout << "magnetic field map uses interpolation mode " << fInterpolation << endl;
}

////////////////////////////////////////////////////////////////////

KMagnetostaticFieldmapCalculator::KMagnetostaticFieldmapCalculator() :
    fSkipExecution(false),
    fOutputFilename(""),
    fDirectory(SCRATCH_DEFAULT_DIR),
    fFile(""),
    fForceUpdate(false),
    fComputeGradient(false),
    fMirrorX(false),
    fMirrorY(false),
    fMirrorZ(false),
    fSpacing(1.)
{}

KMagnetostaticFieldmapCalculator::~KMagnetostaticFieldmapCalculator() = default;

bool KMagnetostaticFieldmapCalculator::CheckPosition(const KPosition& aPosition) const
{
    if (fSpaces.empty())
        return true;

    // check if position is inside ANY space (fails when position is outside ALL spaces)
    // this allows to define multiple spaces and use their logical intersection
    for (const auto* tSpace : fSpaces) {
        if (tSpace->Outside(aPosition) == false)
            return true;
    }
    return false;
}

void KMagnetostaticFieldmapCalculator::Prepare()
{
    fOutputFilename = fDirectory + "/" + fFile;
    if (katrin::KFile::Test(fOutputFilename) && !fForceUpdate) {
        cout << "the vtkImageData file <" << fOutputFilename << "> already exists, skipping calculation" << endl;
        fSkipExecution = true;
        return;
    }

    fSkipExecution = false;

    if (fMagneticFields.empty()) {
        throw KEMSimpleException("KMagnetostaticFieldmap: no magnetic field has been defined.");
        return;
    }

    cout << "initializing magnetic field" << endl;

    for (auto& it : fMagneticFields)
        it.second->Initialize();

    cout << "preparing image data mesh for field map" << endl;

    if ((fLength[0] < 0) || (fLength[1] < 0) || (fLength[2] < 0)) {
        throw KEMSimpleException(
            "KMagnetostaticFieldmapCalculator: invalid grid length: " + std::to_string(fLength.X()) + " m, " +
            std::to_string(fLength.Y()) + " m, " + std::to_string(fLength.Z()) + "m.");
        return;
    }

    if (fSpacing <= 0) {
        throw KEMSimpleException("KMagnetostaticFieldmapCalculator: invalid mesh spacing: " + std::to_string(fSpacing) +
                                 " m");
        return;
    }

    KFieldVector tGridDims =
        KFieldVector(1 + fLength[0] / fSpacing, 1 + fLength[1] / fSpacing, 1 + fLength[2] / fSpacing);
    KFieldVector tGridOrigin = fCenter - 0.5 * fLength;

    if ((ceil(tGridDims[0]) <= 0) || (ceil(tGridDims[1]) <= 0) || (ceil(tGridDims[2]) <= 0)) {
        throw KEMSimpleException(
            "KMagnetostaticFieldmapCalculator: invalid grid dimensions: " + std::to_string(tGridDims.X()) + " m, " +
            std::to_string(tGridDims.Y()) + " m, " + std::to_string(tGridDims.Z()) + "m.");
        return;
    }

    fGrid = vtkSmartPointer<vtkImageData>::New();
    fGrid->SetDimensions((int) ceil(tGridDims[0]), (int) ceil(tGridDims[1]), (int) ceil(tGridDims[2]));
    fGrid->SetOrigin(tGridOrigin[0], tGridOrigin[1], tGridOrigin[2]);
    fGrid->SetSpacing(fSpacing, fSpacing, fSpacing);

    unsigned int tNumPoints = fGrid->GetNumberOfPoints();
    if (tNumPoints < 1) {
        throw KEMSimpleException("invalid number of points: " + std::to_string(tNumPoints));
        return;
    }

    //fieldmsg_debug( "grid has "<<tNumPoints<<" points"<<eom );

    int tDims[3];
    fGrid->GetDimensions(tDims);
    //fieldmsg_debug("grid dimensions are "
    //        <<tDims[0]<<"x"<<tDims[1]<<"x"<<tDims[2]
    //                                              <<eom);

    double tBounds[6];
    fGrid->GetBounds(tBounds);
    //fieldmsg_debug("grid coordinates range from "
    //        <<"("<<tBounds[0]<<"|"<<tBounds[2]<<"|"<<tBounds[4]<<") to "
    //        <<"("<<tBounds[1]<<"|"<<tBounds[3]<<"|"<<tBounds[5]<<")"
    //        <<eom);

    //    fieldmsg_debug("grid center is "
    //            <<"("<<0.5*(tBounds[1]+tBounds[0])<<"|"<<0.5*(tBounds[3]+tBounds[2])<<"|"<<0.5*(tBounds[5]+tBounds[4])<<")"
    //            <<eom);

    if (fMirrorX || fMirrorY || fMirrorZ) {
        cout << "mirroring points along " << (fMirrorX ? "x" : "") << (fMirrorY ? "y" : "") << (fMirrorZ ? "z" : "")
             << "-axis, effective number of points reduced to "
             << tNumPoints / ((fMirrorX ? 2 : 1) * (fMirrorY ? 2 : 1) * (fMirrorZ ? 2 : 1)) << endl;
    }

    fValidityData = vtkSmartPointer<vtkIntArray>::New();
    fValidityData->SetName("validity");
    fValidityData->SetNumberOfComponents(1);  // scalar data
    fValidityData->SetNumberOfTuples(tNumPoints);
    fGrid->GetPointData()->AddArray(fValidityData);

    fFieldData = vtkSmartPointer<vtkDoubleArray>::New();
    fFieldData->SetName("magnetic field");
    fFieldData->SetNumberOfComponents(3);  // vector data
    fFieldData->SetNumberOfTuples(tNumPoints);
    fGrid->GetPointData()->AddArray(fFieldData);

    if (fComputeGradient) {
        if (fMagneticFields.size() > 1) {
            cout << "computing magnetic gradient of multiple fields, results are probably incorrect" << endl;
        }

        fGradientData = vtkSmartPointer<vtkDoubleArray>::New();
        fGradientData->SetName("magnetic gradient");
        fGradientData->SetNumberOfComponents(9);  // tensor data
        fGradientData->SetNumberOfTuples(tNumPoints);
        fGrid->GetPointData()->AddArray(fGradientData);
    }
}

void KMagnetostaticFieldmapCalculator::Execute()
{
    if (fSkipExecution)
        return;

    fValidityData->FillComponent(0, 0);  // initialize all to 0 = invalid

    unsigned int tNumPoints = fGrid->GetNumberOfPoints();

    //timer
    clock_t tClockStart, tClockEnd;
    double tTimeSpent;


    cout << "computing magnetic field at " << tNumPoints << " grid points" << endl;

    //evaluate field
    tClockStart = clock();
    for (unsigned int i = 0; i < tNumPoints; i++) {
        if (i % 10 == 0) {
            int progress = 50 * (float) i / (float) (tNumPoints - 1);
            std::cout << "\r  ";
            for (int j = 0; j < 50; j++)
                std::cout << (j <= progress ? "#" : ".");
            std::cout << "  [" << 2 * progress << "%]" << std::flush;
        }

        double tPoint[3];
        fGrid->GetPoint(i, tPoint);

        if (!CheckPosition(KPosition(tPoint)))
            continue;

        bool tHasValue = false;
        KFieldVector tField;

        if (fMirrorX || fMirrorY || fMirrorZ) {
            double tMirrorPoint[3];
            tMirrorPoint[0] = tPoint[0];
            tMirrorPoint[1] = tPoint[1];
            tMirrorPoint[2] = tPoint[2];
            if (fMirrorX && (tPoint[0] > fCenter.X()))
                tMirrorPoint[0] = 2. * fCenter.X() - tPoint[0];
            if (fMirrorY && (tPoint[1] > fCenter.Y()))
                tMirrorPoint[1] = 2. * fCenter.Y() - tPoint[1];
            if (fMirrorZ && (tPoint[2] > fCenter.Z()))
                tMirrorPoint[2] = 2. * fCenter.Z() - tPoint[2];

            if ((tMirrorPoint[0] != tPoint[0]) || (tMirrorPoint[1] != tPoint[1]) || (tMirrorPoint[2] != tPoint[2])) {
                unsigned int j = fGrid->FindPoint(tMirrorPoint);
                if (fValidityData->GetTuple1(j) >= 2)  // 2 = field valid
                {
                    tField.SetComponents(fFieldData->GetTuple3(j));
                    tHasValue = true;
                }
            }
        }

        if (!tHasValue) {
            tField = KFieldVector::sZero;
            try {
                for (auto& it : fMagneticFields)
                    tField += it.second->MagneticField(KPosition(tPoint));
            }
            catch (katrin::KGslException& e) {
                tField = KFieldVector::sInvalid;
            }
        }

        fFieldData->SetTuple3(i, tField[0], tField[1], tField[2]);
        fValidityData->SetTuple1(i, 2);
    }
    std::cout << std::endl;
    tClockEnd = clock();

    tTimeSpent = ((double) (tClockEnd - tClockStart)) / CLOCKS_PER_SEC;  // time in seconds
    cout << "finished computing field map (total time spent = " << tTimeSpent
         << ", time per field evaluation = " << tTimeSpent / (double) (tNumPoints) << ")" << endl;


    if (fComputeGradient) {
        cout << "computing magnetic gradient at " << tNumPoints << " grid points" << endl;

        //evaluate field
        tClockStart = clock();
        for (unsigned int i = 0; i < tNumPoints; i++) {
            if (i % 10 == 0) {
                int progress = 50 * (float) i / (float) (tNumPoints - 1);
                std::cout << "\r  ";
                for (int j = 0; j < 50; j++)
                    std::cout << (j <= progress ? "#" : ".");
                std::cout << "  [" << 2 * progress << "%]" << std::flush;
            }

            double tPoint[3];
            fGrid->GetPoint(i, tPoint);

            if (!CheckPosition(KPosition(tPoint)))
                continue;

            bool tHasValue = false;
            KThreeMatrix tGradient;

            if (fMirrorX || fMirrorY || fMirrorZ) {
                double tMirrorPoint[3];
                tMirrorPoint[0] = tPoint[0];
                tMirrorPoint[1] = tPoint[1];
                tMirrorPoint[2] = tPoint[2];
                if (fMirrorX && (tPoint[0] > fCenter.X()))
                    tMirrorPoint[0] = 2. * fCenter.X() - tPoint[0];
                if (fMirrorY && (tPoint[1] > fCenter.Y()))
                    tMirrorPoint[1] = 2. * fCenter.Y() - tPoint[1];
                if (fMirrorZ && (tPoint[2] > fCenter.Z()))
                    tMirrorPoint[2] = 2. * fCenter.Z() - tPoint[2];

                if ((tMirrorPoint[0] != tPoint[0]) || (tMirrorPoint[1] != tPoint[1]) ||
                    (tMirrorPoint[2] != tPoint[2])) {
                    unsigned int j = fGrid->FindPoint(tMirrorPoint);
                    if (fValidityData->GetTuple1(j) >= 3)  // 3 = gradient valid
                    {
                        tGradient.SetComponents(fGradientData->GetTuple9(j));
                        tHasValue = true;
                    }
                }
            }

            if (!tHasValue) {
                tGradient = KThreeMatrix::sZero;
                try {
                    /// FIXME: summing up gradients from multiple fields doesn't make much sense,
                    /// Needs to be handled in a better way, e.g. calculate gradient from field map directly.
                    for (auto& it : fMagneticFields)
                        tGradient += it.second->MagneticGradient(KPosition(tPoint));
                }
                catch (katrin::KGslException& e) {
                    tGradient = KThreeMatrix::sInvalid;
                }
            }

            fGradientData->SetTuple9(i,
                                     tGradient[0],
                                     tGradient[1],
                                     tGradient[2],
                                     tGradient[3],
                                     tGradient[4],
                                     tGradient[5],
                                     tGradient[6],
                                     tGradient[7],
                                     tGradient[8]);
            fValidityData->SetTuple1(i, 3);
        }
        std::cout << std::endl;
        tClockEnd = clock();

        tTimeSpent = ((double) (tClockEnd - tClockStart)) / CLOCKS_PER_SEC;  // time in seconds
        cout << "finished computing gradient map (total time spent = " << tTimeSpent
             << ", time per gradient evaluation = " << tTimeSpent / (double) (tNumPoints) << ")" << endl;
    }
    else {
        cout << "not computing magnetic gradient" << endl;
    }
}

void KMagnetostaticFieldmapCalculator::Finish()
{
    if (fSkipExecution)
        return;

    cout << "exporting vtkImageData file <" << fOutputFilename << ">" << endl;

    vtkSmartPointer<vtkXMLImageDataWriter> vWriter = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    vWriter->SetFileName(fOutputFilename.c_str());
#if (VTK_MAJOR_VERSION >= 6)
    vWriter->SetInputData(fGrid);
#else
    vWriter->SetInput(fGrid);
#endif
    vWriter->SetDataModeToBinary();
    vWriter->Write();

    cout << "finished writing vtkImageData file" << endl;

    return;
}

void KMagnetostaticFieldmapCalculator::Initialize()
{
    Prepare();
    Execute();
    Finish();
}


} /* namespace KEMField */
