/*
 * KElectrostaticPotentialMap.cc
 *
 *  Created on: 26 Apr 2016
 *      Author: wolfgang
 */

#include "KElectrostaticPotentialMap.hh"

#include "KFile.h"
#include "KEMFileInterface.hh"
#include "KEMVectorConverters.hh"

#include "KEMCout.hh"
#include "KEMSimpleException.hh"

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>

namespace KEMField {



KPotentialMapVTK::KPotentialMapVTK( const string& aFilename )
{
    cout << "loading potential map from file <" << aFilename << ">" << endl;

    vtkXMLImageDataReader* reader = vtkXMLImageDataReader::New();
    reader->SetFileName( aFilename.c_str() );
    reader->Update();
    fImageData = reader->GetOutput();

    int dims[3];
    double bounds[6];
    fImageData->GetDimensions(dims);
    fImageData->GetBounds(bounds);
    //fieldmsg_debug( "potential map has " << fImageData->GetNumberOfPoints() << " points (" << dims[0] << "x" << dims[1] << "x" << dims[2] << ") and ranges from " << KThreeVector(bounds[0],bounds[2],bounds[4]) << " to " << KThreeVector(bounds[1],bounds[3],bounds[4]) << eom);
}

KPotentialMapVTK::~KPotentialMapVTK()
{
}

bool KPotentialMapVTK::GetValue( const string& array, const KPosition& aSamplePoint, double *aValue ) const
{
    vtkDataArray *data = fImageData->GetPointData()->GetArray( array.c_str() );

    // get coordinates of closest mesh point
    vtkIdType center = fImageData->FindPoint( (double*)(aSamplePoint.Components()) );
    if (center < 0)
        return false;

    // get value at center
    data->GetTuple( center, aValue );

    return true;
}

bool KPotentialMapVTK::GetPotential( const KPosition& aSamplePoint, const double& /*aSampleTime*/, double& aPotential ) const
{
    //fieldmsg_debug( "sampling electric potential at point " << aSamplePoint << eom);

    double value;
    if ( GetValue( "electric potential", aSamplePoint, &value ) )
    {
        aPotential = value;
        return true;
    }
    return false;
}

bool KPotentialMapVTK::GetField( const KPosition& aSamplePoint, const double& /*aSampleTime*/, KEMThreeVector& aField ) const
{
    //fieldmsg_debug( "sampling electric field at point " << aSamplePoint << eom);

    double value[3];
    if ( GetValue( "electric field", aSamplePoint, value ) )
    {
        aField.SetComponents( value );
        return true;
    }
    return false;
}

KLinearInterpolationPotentialMapVTK::KLinearInterpolationPotentialMapVTK( const string& aFilename ) :
                KPotentialMapVTK( aFilename )
{
}

KLinearInterpolationPotentialMapVTK::~KLinearInterpolationPotentialMapVTK()
{
}

bool KLinearInterpolationPotentialMapVTK::GetValue( const string& array, const KPosition& aSamplePoint, double *aValue ) const
{
    vtkDataArray *data = fImageData->GetPointData()->GetArray( array.c_str() );

    // get coordinates of surrounding mesh points
    static const char map[8][3] =
    {
            { -1, -1, -1 },  // c000
            {  1, -1, -1 },  // c100
            { -1,  1, -1 },  // c010
            {  1,  1, -1 },  // c110
            { -1, -1,  1 },  // c001
            {  1, -1,  1 },  // c101
            { -1,  1,  1 },  // c011
            {  1,  1,  1 },  // c111
    };
    static KEMThreeVector vertices[8];
    static double values[3][8];  // always allocate for vectors even if we have scalars (to be safe) - note that array ordering is swapped

    double *spacing = fImageData->GetSpacing();
    for ( int i = 0; i < 8; i++ )
    {
        // first compute the coordinates of the surrounding mesh points ...
        KEMThreeVector point = aSamplePoint + 0.5 * KEMThreeVector( map[i][0]*spacing[0], map[i][1]*spacing[1], map[i][2]*spacing[2] );
        vtkIdType corner = fImageData->FindPoint( (double*)(point.Components()) );
        if (corner < 0)
            return false;
        // ... then retrieve data at these points
        vertices[i] = fImageData->GetPoint( corner );
        for (int k = 0; k < data->GetNumberOfComponents(); k++ )
            values[k][i] = data->GetComponent( corner, k);
    }

    // get interpolated value at center
    double xd = (aSamplePoint.X() - vertices[0][0]) / spacing[0];
    double yd = (aSamplePoint.Y() - vertices[0][1]) / spacing[1];
    double zd = (aSamplePoint.Z() - vertices[0][2]) / spacing[2];
    for (int k = 0; k < data->GetNumberOfComponents();  k++ )
    {
        double c00 = values[k][0] * (1 - xd) + values[k][1] * xd;
        double c10 = values[k][2] * (1 - xd) + values[k][3] * xd;
        double c01 = values[k][4] * (1 - xd) + values[k][5] * xd;
        double c11 = values[k][6] * (1 - xd) + values[k][7] * xd;

        double c0 = c00 * (1 - yd) + c10 * yd;
        double c1 = c01 * (1 - yd) + c11 * yd;

        double c = c0 * (1 - zd) + c1 * zd;

        aValue[k] = c;
    }

    return true;
}

KCubicInterpolationPotentialMapVTK::KCubicInterpolationPotentialMapVTK( const string& aFilename ) :
                KPotentialMapVTK( aFilename )
{
}

KCubicInterpolationPotentialMapVTK::~KCubicInterpolationPotentialMapVTK()
{
}

bool KCubicInterpolationPotentialMapVTK::GetValue( const string& array, const KPosition& aSamplePoint, double *aValue ) const
{
    vtkDataArray *data = fImageData->GetPointData()->GetArray( array.c_str() );

    // get coordinates of surrounding mesh points
    static const char map[64][3] =
    {
            //
            { -3, -3, -3 },  // 00
            { -3, -3, -1 },
            { -3, -3,  1 },
            { -3, -3,  3 },

            { -3, -1, -3 },  // 04
            { -3, -1, -1 },
            { -3, -1,  1 },
            { -3, -1,  3 },

            { -3,  1, -3 },  // 08
            { -3,  1, -1 },
            { -3,  1,  1 },
            { -3,  1,  3 },

            { -3,  3, -3 },  // 12
            { -3,  3, -1 },
            { -3,  3,  1 },
            { -3,  3,  3 },
            //
            { -1, -3, -3 },  // 16
            { -1, -3, -1 },
            { -1, -3,  1 },
            { -1, -3,  3 },

            { -1, -1, -3 },  // 20
            { -1, -1, -1 },
            { -1, -1,  1 },
            { -1, -1,  3 },

            { -1,  1, -3 },  // 24
            { -1,  1, -1 },
            { -1,  1,  1 },
            { -1,  1,  3 },

            { -1,  3, -3 },  // 28
            { -1,  3, -1 },
            { -1,  3,  1 },
            { -1,  3,  3 },
            //
            {  1, -3, -3 },  // 32
            {  1, -3, -1 },
            {  1, -3,  1 },
            {  1, -3,  3 },

            {  1, -1, -3 },  // 36
            {  1, -1, -1 },
            {  1, -1,  1 },
            {  1, -1,  3 },

            {  1,  1, -3 },  // 40
            {  1,  1, -1 },
            {  1,  1,  1 },
            {  1,  1,  3 },

            {  1,  3, -3 },  // 44
            {  1,  3, -1 },
            {  1,  3,  1 },
            {  1,  3,  3 },
            //
            {  3, -3, -3 },  // 48
            {  3, -3, -1 },
            {  3, -3,  1 },
            {  3, -3,  3 },

            {  3, -1, -3 },  // 52
            {  3, -1, -1 },
            {  3, -1,  1 },
            {  3, -1,  3 },

            {  3,  1, -3 },  // 56
            {  3,  1, -1 },
            {  3,  1,  1 },
            {  3,  1,  3 },

            {  3,  3, -3 },  // 60
            {  3,  3, -1 },
            {  3,  3,  1 },
            {  3,  3,  3 },
    };
    static KEMThreeVector vertices[64];
    static double values[3][64];  // always allocate for vectors even if we have scalars (to be safe) - note that array ordering is swapped

    double *spacing = fImageData->GetSpacing();
    for ( int i = 0; i < 64; i++ )
    {
        // first compute the coordinates of the surrounding mesh points ...
        KEMThreeVector point = aSamplePoint + 0.5 * KEMThreeVector( map[i][0]*spacing[0], map[i][1]*spacing[1], map[i][2]*spacing[2] );
        vtkIdType corner = fImageData->FindPoint( (double*)(point.Components()) );
        if (corner < 0)
            return false;
        // ... then retrieve data at these points
        vertices[i] = fImageData->GetPoint( corner );
        for (int k = 0; k < data->GetNumberOfComponents(); k++ )
            values[k][i] = data->GetComponent( corner, k);
    }

    double xd = (aSamplePoint.X() - vertices[21][0]) / spacing[0];  // point index 21 is at -1/-1/-1 coords = "lower" corner
    double yd = (aSamplePoint.Y() - vertices[21][1]) / spacing[1];
    double zd = (aSamplePoint.Z() - vertices[21][2]) / spacing[2];
    for (int k = 0; k < data->GetNumberOfComponents(); k++ )
    {
        aValue[k] = _tricubicInterpolate( &(values[k][0]), xd, yd, zd );
    }

    return true;
}

double KCubicInterpolationPotentialMapVTK::_cubicInterpolate (double p[], double x)  // array of 4
{
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.*p[0] - 5.*p[1] + 4.*p[2] - p[3] + x*(3.*(p[1] - p[2]) + p[3] - p[0])));
}

double KCubicInterpolationPotentialMapVTK::_bicubicInterpolate (double p[], double x, double y)  // array of 4x4
{
    static double q[4];
    q[0] = _cubicInterpolate(&(p[0]), y);
    q[1] = _cubicInterpolate(&(p[4]), y);
    q[2] = _cubicInterpolate(&(p[8]), y);
    q[3] = _cubicInterpolate(&(p[12]), y);
    return _cubicInterpolate(q, x);
}

double KCubicInterpolationPotentialMapVTK::_tricubicInterpolate (double p[], double x, double y, double z)  // array of 4x4x4
{
    static double q[4];
    q[0] = _bicubicInterpolate(&(p[0]), y, z);
    q[1] = _bicubicInterpolate(&(p[16]), y, z);
    q[2] = _bicubicInterpolate(&(p[32]), y, z);
    q[3] = _bicubicInterpolate(&(p[48]), y, z);
    return _cubicInterpolate(q, x);
}
////////////////////////////////////////////////////////////////////

KElectrostaticPotentialmap::KElectrostaticPotentialmap() :
                fDirectory( SCRATCH_DEFAULT_DIR ),
                fFile(),
                fInterpolation( 0 ),
                fPotentialMap( NULL )
{
}

KElectrostaticPotentialmap::~KElectrostaticPotentialmap()
{
}

double KElectrostaticPotentialmap::PotentialCore( const KPosition& P) const
{
    double tPotential = 0;
    double aRandomTime = 0;
    if (! fPotentialMap->GetPotential( P, aRandomTime, tPotential ))
        cout << "WARNING: could not compute electric potential at sample point " << P << endl;

    return tPotential;
}

KEMThreeVector KElectrostaticPotentialmap::ElectricFieldCore( const KPosition& P) const
{
    KEMThreeVector tField;
    tField.SetComponents(0.,0.,0.);
    double aRandomTime = 0;
    if (! fPotentialMap->GetField( P, aRandomTime, tField ))
        cout << "WARNING: could not compute electric field at sample point " << P << endl;

    return tField;
}

void KElectrostaticPotentialmap::SetDirectory( const std::string& aDirectory )
{
    fDirectory = aDirectory;
    return;
}

void KElectrostaticPotentialmap::SetFile( const std::string& aFile )
{
    fFile = aFile;
    return;
}

void KElectrostaticPotentialmap::SetInterpolation( const string& aMode )
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

void KElectrostaticPotentialmap::InitializeCore()
{
    string filename = fDirectory + "/" + fFile;

    /// one could use different data back-ends here (e.g. ROOT instead of VTK, or ASCII files ...)
    switch ( fInterpolation )
    {
    case 0:
        fPotentialMap = new KPotentialMapVTK( filename );
        break;
    case 1:
        fPotentialMap = new KLinearInterpolationPotentialMapVTK( filename );
        break;
    case 3:
        fPotentialMap = new KCubicInterpolationPotentialMapVTK( filename );
        break;
    default:
        throw KEMSimpleException( "interpolation mode " + std::to_string(fInterpolation) + " is not implemented");
        break;
    }

    cout << "electric potential map uses interpolation mode " << fInterpolation << endl;
}

////////////////////////////////////////////////////////////////////

KElectrostaticPotentialmapCalculator::KElectrostaticPotentialmapCalculator() :
                fDirectory( SCRATCH_DEFAULT_DIR ),
                fFile( "" ),
                fCenter(),
                fLength(),
                fMirrorX( false ),
                fMirrorY( false ),
                fMirrorZ( false ),
                fSpacing( 1. ),
                fElectricField( NULL )
{
}

KElectrostaticPotentialmapCalculator::~KElectrostaticPotentialmapCalculator()
{
}

bool KElectrostaticPotentialmapCalculator::CheckPosition( const KPosition& aPosition ) const
{
    if (fSpaces.size() == 0)
        return true;

    // check if position is inside ANY space (fails when position is outside ALL spaces)
    // this allows to define multiple spaces and use their logical intersection
    for ( auto tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); ++tSpaceIt )
    {
        const KGeoBag::KGSpace *tSpace = (*tSpaceIt);
        if ( tSpace->Outside( KEM2KThreeVector(aPosition) ) == false )
            return true;
    }
    return false;
}

void KElectrostaticPotentialmapCalculator::Prepare()
{
    if (! fElectricField)
    {
        throw KEMSimpleException("KElectrostaticPotentialMap: no electric field has been defined.");
        return;
    }

    cout << "initializing electric field" << endl;

    fElectricField->Initialize();


    cout << "preparing image data mesh for potential map" << endl;

    if ((fLength[0] < 0) || (fLength[1] < 0) || (fLength[2] < 0))
    {
        throw KEMSimpleException("KEletrostaticPotentialmapCalculator: invalid grid length: "
                + std::to_string(fLength.X()) + " m, "
                + std::to_string(fLength.Y()) + " m, "
                + std::to_string(fLength.Z()) + "m." );
        return;
    }

    if (fSpacing <= 0)
    {
        throw KEMSimpleException("KElectrostaticPotentialmapCalculator: invalid mesh spacing: "
                + std::to_string(fSpacing) + " m" );
        return;
    }

    KEMThreeVector tGridDims = KEMThreeVector( 1 + fLength[0]/fSpacing, 1 + fLength[1]/fSpacing, 1 + fLength[2]/fSpacing );
    KEMThreeVector tGridOrigin = fCenter - 0.5*fLength;

    if ((floor(tGridDims[0]) <= 0) || (floor(tGridDims[1]) <= 0) || (floor(tGridDims[2]) <= 0))
    {
        throw KEMSimpleException( "KElectrostaticPotentialmapCalculator: invalid grid dimensions: "
                + std::to_string(tGridDims.X()) + " m, "
                + std::to_string(tGridDims.Y()) + " m, "
                + std::to_string(tGridDims.Z()) + "m." );
        return;
    }

    fGrid = vtkSmartPointer<vtkImageData>::New();
    fGrid->SetDimensions( (int)floor(tGridDims[0]), (int)floor(tGridDims[1]), (int)floor(tGridDims[2]) );
    fGrid->SetOrigin( tGridOrigin[0], tGridOrigin[1], tGridOrigin[2] );
    fGrid->SetSpacing( fSpacing, fSpacing, fSpacing );

    unsigned int tNumPoints = fGrid->GetNumberOfPoints();
    if (tNumPoints < 1)
    {
        throw KEMSimpleException( "invalid number of points: " + std::to_string(tNumPoints) );
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

    if (fMirrorX || fMirrorY || fMirrorZ)
    {
        cout << "mirroring points along "
                <<(fMirrorX ? "x" : "")
                <<(fMirrorY ? "y" : "")
                <<(fMirrorZ ? "z" : "")
                <<"-axis, effective number of points reduced to "
                <<tNumPoints/((fMirrorX?2:1)*(fMirrorY?2:1)*(fMirrorZ?2:1))
                <<endl;
    }

    fValidityData = vtkSmartPointer<vtkIntArray>::New();
    fValidityData->SetName("validity");
    fValidityData->SetNumberOfComponents(1);  // scalar data
    fValidityData->SetNumberOfTuples(tNumPoints);
    fGrid->GetPointData()->AddArray(fValidityData);

    fPotentialData = vtkSmartPointer<vtkDoubleArray>::New();
    fPotentialData->SetName("electric potential");
    fPotentialData->SetNumberOfComponents(1);  // scalar data
    fPotentialData->SetNumberOfTuples(tNumPoints);
    fGrid->GetPointData()->AddArray(fPotentialData);

    fFieldData = vtkSmartPointer<vtkDoubleArray>::New();
    fFieldData->SetName("electric field");
    fFieldData->SetNumberOfComponents(3);  // vector data
    fFieldData->SetNumberOfTuples(tNumPoints);
    fGrid->GetPointData()->AddArray(fFieldData);
}

void KElectrostaticPotentialmapCalculator::Execute()
{
    fValidityData->FillComponent(0, 0);  // initialize all to 0 = invalid

    unsigned int tNumPoints = fGrid->GetNumberOfPoints();

    //timer
    clock_t tClockStart, tClockEnd;
    double tTimeSpent;


    cout << "computing electric potential at " << tNumPoints << " grid points" << endl;

    //evaluate potential
    tClockStart = clock();
    for (unsigned int i = 0; i < tNumPoints; i++)
    {
        if (i % 10 == 0)
        {
            int progress = 50*(float)i / (float)(tNumPoints-1);
            std::cout<<"\r  ";
            for(int j=0; j<50; j++)
                std::cout<<(j<=progress?"#":".");
            std::cout<<"  ["<<2*progress<<"%]"<<std::flush;
        }

        double tPoint[3];
        fGrid->GetPoint(i, tPoint);

        if (! CheckPosition(KPosition(tPoint)) )
            continue;

        bool tHasValue = false;
        double tPotential = 0.;

        if (fMirrorX || fMirrorY || fMirrorZ)
        {
            double tMirrorPoint[3];
            tMirrorPoint[0] = tPoint[0];
            tMirrorPoint[1] = tPoint[1];
            tMirrorPoint[2] = tPoint[2];
            if (fMirrorX && (tPoint[0] > fCenter.X()))
                tMirrorPoint[0] = 2.*fCenter.X() - tPoint[0];
            if (fMirrorY && (tPoint[1] > fCenter.Y()))
                tMirrorPoint[1] = 2.*fCenter.Y() - tPoint[1];
            if (fMirrorZ && (tPoint[2] > fCenter.Z()))
                tMirrorPoint[2] = 2.*fCenter.Z() - tPoint[2];

            if ((tMirrorPoint[0] != tPoint[0]) || (tMirrorPoint[1] != tPoint[1]) || (tMirrorPoint[2] != tPoint[2]))
            {
                unsigned int j = fGrid->FindPoint(tMirrorPoint);
                if (fValidityData->GetTuple1(j) >= 1)  // 1 = potential valid
                        {
                    tPotential = fPotentialData->GetTuple1(j);
                    tHasValue = true;
                        }
            }
        }

        if (! tHasValue)
        {
            tPotential = fElectricField->Potential(KEMThreeVector(tPoint));
        }

        fPotentialData->SetTuple1(i, tPotential);
        fValidityData->SetTuple1(i, 1);
    }
    std::cout<<std::endl;
    tClockEnd = clock();

    tTimeSpent = ((double)(tClockEnd - tClockStart))/CLOCKS_PER_SEC; // time in seconds
    cout << "finished computing potential map (total time spent = " << tTimeSpent << ", time per potential evaluation = " << tTimeSpent/(double)(tNumPoints) << ")" << endl;


    cout << "computing electric field at " << tNumPoints << " grid points" << endl;

    //evaluate field
    tClockStart = clock();
    for (unsigned int i = 0; i < tNumPoints; i++)
    {
        if (i % 10 == 0)
        {
            int progress = 50*(float)i / (float)(tNumPoints-1);
            std::cout<<"\r  ";
            for(int j=0; j<50; j++)
                std::cout<<(j<=progress?"#":".");
            std::cout<<"  ["<<2*progress<<"%]"<<std::flush;
        }

        double tPoint[3];
        fGrid->GetPoint(i, tPoint);

        if (! CheckPosition(KPosition(tPoint)) )
            continue;

        bool tHasValue = false;
        KEMThreeVector tField;

        if (fMirrorX || fMirrorY || fMirrorZ)
        {
            double tMirrorPoint[3];
            tMirrorPoint[0] = tPoint[0];
            tMirrorPoint[1] = tPoint[1];
            tMirrorPoint[2] = tPoint[2];
            if (fMirrorX && (tPoint[0] > fCenter.X()))
                tMirrorPoint[0] = 2.*fCenter.X() - tPoint[0];
            if (fMirrorY && (tPoint[1] > fCenter.Y()))
                tMirrorPoint[1] = 2.*fCenter.Y() - tPoint[1];
            if (fMirrorZ && (tPoint[2] > fCenter.Z()))
                tMirrorPoint[2] = 2.*fCenter.Z() - tPoint[2];

            if ((tMirrorPoint[0] != tPoint[0]) || (tMirrorPoint[1] != tPoint[1]) || (tMirrorPoint[2] != tPoint[2]))
            {
                unsigned int j = fGrid->FindPoint(tMirrorPoint);
                if (fValidityData->GetTuple1(j) >= 2)  // 2 = field valid
                        {
                    tField.SetComponents( fFieldData->GetTuple3(j) );
                    tHasValue = true;
                        }
            }
        }

        if (! tHasValue)
        {
            tField = fElectricField->ElectricField(KPosition(tPoint));
        }

        fFieldData->SetTuple3(i, tField[0], tField[1], tField[2]);
        fValidityData->SetTuple1(i, 2);
    }
    std::cout<<std::endl;
    tClockEnd = clock();

    tTimeSpent = ((double)(tClockEnd - tClockStart))/CLOCKS_PER_SEC; // time in seconds
    cout << "finished computing potential map (total time spent = " << tTimeSpent << ", time per field evaluation = " << tTimeSpent/(double)(tNumPoints) << ")" << endl;
}

void KElectrostaticPotentialmapCalculator::Finish()
{
    string filename = fDirectory + "/" + fFile;
    cout << "exporting vtkImageData file <" << filename << ">" << endl;

    vtkSmartPointer<vtkXMLImageDataWriter> vWriter = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    vWriter->SetFileName(filename.c_str());
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

void KElectrostaticPotentialmapCalculator::Initialize()
{
    Prepare();
    Execute();
    Finish();
}


} /* namespace KEMField */
