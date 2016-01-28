#include "KSFieldElectricPotentialmap.h"
#include "KSFieldsMessage.h"

#include "KFile.h"
#include "KEMFileInterface.hh"

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>

namespace Kassiopeia
{

    KPotentialMapVTK::KPotentialMapVTK( const string& aFilename )
    {
        fieldmsg(eNormal) << "loading potential map from file <" << aFilename << ">" << eom;

        vtkXMLImageDataReader* reader = vtkXMLImageDataReader::New();
        reader->SetFileName( aFilename.c_str() );
        reader->Update();
        fImageData = reader->GetOutput();

        int dims[3];
        double bounds[6];
        fImageData->GetDimensions(dims);
        fImageData->GetBounds(bounds);
        fieldmsg_debug( "potential map has " << fImageData->GetNumberOfPoints() << " points (" << dims[0] << "x" << dims[1] << "x" << dims[2] << ") and ranges from " << KThreeVector(bounds[0],bounds[2],bounds[4]) << " to " << KThreeVector(bounds[1],bounds[3],bounds[4]) << eom);
    }

    KPotentialMapVTK::~KPotentialMapVTK()
    {
    }

    bool KPotentialMapVTK::GetValue( const string& array, const KThreeVector& aSamplePoint, double *aValue )
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

    bool KPotentialMapVTK::GetPotential( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, double& aPotential )
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

    bool KPotentialMapVTK::GetField( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aField )
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

    bool KLinearInterpolationPotentialMapVTK::GetValue( const string& array, const KThreeVector& aSamplePoint, double *aValue )
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
        static KThreeVector vertices[8];
        static double values[3][8];  // always allocate for vectors even if we have scalars (to be safe) - note that array ordering is swapped

        double *spacing = fImageData->GetSpacing();
        for ( int i = 0; i < 8; i++ )
        {
            // first compute the coordinates of the surrounding mesh points ...
            KThreeVector point = aSamplePoint + 0.5 * KThreeVector( map[i][0]*spacing[0], map[i][1]*spacing[1], map[i][2]*spacing[2] );
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

    bool KCubicInterpolationPotentialMapVTK::GetValue( const string& array, const KThreeVector& aSamplePoint, double *aValue )
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
        static KThreeVector vertices[64];
        static double values[3][64];  // always allocate for vectors even if we have scalars (to be safe) - note that array ordering is swapped

        double *spacing = fImageData->GetSpacing();
        for ( int i = 0; i < 64; i++ )
        {
            // first compute the coordinates of the surrounding mesh points ...
            KThreeVector point = aSamplePoint + 0.5 * KThreeVector( map[i][0]*spacing[0], map[i][1]*spacing[1], map[i][2]*spacing[2] );
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

    KSFieldElectricPotentialmap::KSFieldElectricPotentialmap() :
        fDirectory( SCRATCH_DEFAULT_DIR ),
        fFile(),
        fInterpolation( 0 ),
        fPotentialMap( NULL )
    {
    }
    KSFieldElectricPotentialmap::KSFieldElectricPotentialmap( const KSFieldElectricPotentialmap& aCopy ) :
        KSComponent(),
        fDirectory( aCopy.fDirectory ),
        fFile( aCopy.fFile ),
        fPotentialMap( aCopy.fPotentialMap )
    {
    }
    KSFieldElectricPotentialmap* KSFieldElectricPotentialmap::Clone() const
    {
        return new KSFieldElectricPotentialmap( *this );
    }
    KSFieldElectricPotentialmap::~KSFieldElectricPotentialmap()
    {
    }

    void KSFieldElectricPotentialmap::CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential )
    {
        aPotential = 0.;
        double tPotential;
        if (! fPotentialMap->GetPotential( aSamplePoint, aSampleTime, tPotential ))
            fieldmsg( eWarning ) << "could not compute electric potential at sample point " << aSamplePoint << eom;

        aPotential = tPotential;
        return;
    }

    void KSFieldElectricPotentialmap::CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField )
    {
        aField.SetComponents(0.,0.,0.);
        KThreeVector tField;
        if (! fPotentialMap->GetField( aSamplePoint, aSampleTime, tField ))
            fieldmsg( eWarning ) << "could not compute electric field at sample point " << aSamplePoint << eom;

        aField = tField;
        return;
    }

    void KSFieldElectricPotentialmap::SetDirectory( const std::string& aDirectory )
    {
        fDirectory = aDirectory;
        return;
    }

    void KSFieldElectricPotentialmap::SetFile( const std::string& aFile )
    {
        fFile = aFile;
        return;
    }

    void KSFieldElectricPotentialmap::SetInterpolation( const string& aMode )
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

    void KSFieldElectricPotentialmap::InitializeComponent()
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
                fieldmsg( eError ) << "interpolation mode " << fInterpolation << " is not implemented" << eom;
                break;
        }

        fieldmsg( eNormal) << "electric potential map uses interpolation mode " << fInterpolation << eom;
    }

    void KSFieldElectricPotentialmap::DeinitializeComponent()
    {
        delete fPotentialMap;
        fPotentialMap = NULL;
    }

    ////////////////////////////////////////////////////////////////////

    KSFieldElectricPotentialmapCalculator::KSFieldElectricPotentialmapCalculator() :
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
    KSFieldElectricPotentialmapCalculator::KSFieldElectricPotentialmapCalculator( const KSFieldElectricPotentialmapCalculator& aCopy ) :
        KSComponent(),
        fDirectory( aCopy.fDirectory ),
        fFile( aCopy.fFile ),
        fCenter( aCopy.fCenter ),
        fLength( aCopy.fLength ),
        fMirrorX( aCopy.fMirrorX ),
        fMirrorY( aCopy.fMirrorY ),
        fMirrorZ( aCopy.fMirrorZ ),
        fSpacing( aCopy.fSpacing ),
        fElectricField( aCopy.fElectricField )
    {
    }
    KSComponent* KSFieldElectricPotentialmapCalculator::Clone() const
    {
        return new KSFieldElectricPotentialmapCalculator ( *this );
    }

    KSFieldElectricPotentialmapCalculator::~KSFieldElectricPotentialmapCalculator()
    {
    }

    bool KSFieldElectricPotentialmapCalculator::CheckPosition( const KThreeVector& aPosition ) const
    {
        if (fSpaces.size() == 0)
            return true;

        // check if position is inside ANY space (fails when position is outside ALL spaces)
        // this allows to define multiple spaces and use their logical disjunction
        for ( auto tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); ++tSpaceIt )
        {
            const KGeoBag::KGSpace *tSpace = (*tSpaceIt);
            if ( tSpace->Outside( aPosition ) == false )
                return true;
        }
        return false;
    }

    bool KSFieldElectricPotentialmapCalculator::Prepare()
    {
        if (! fElectricField)
        {
            fieldmsg( eError ) << "no electric field has been defined" << eom;
            return false;
        }

        fieldmsg( eNormal ) << "preparing electric field <" << fElectricField->GetName() << ">" << eom;

        fElectricField->Initialize();


        fieldmsg( eNormal ) << "preparing image data mesh for potential map" << eom;

        if ((fLength[0] < 0) || (fLength[1] < 0) || (fLength[2] < 0))
        {
            fieldmsg( eError ) << "invalid grid length: " << fLength << " m" << eom;
            return false;
        }

        if (fSpacing <= 0)
        {
            fieldmsg( eError ) << "invalid mesh spacing: " << fSpacing << " m" << eom;
            return false;
        }

        KThreeVector tGridDims = KThreeVector( 1 + fLength[0]/fSpacing, 1 + fLength[1]/fSpacing, 1 + fLength[2]/fSpacing );
        KThreeVector tGridOrigin = fCenter - 0.5*fLength;

        if ((floor(tGridDims[0]) <= 0) || (floor(tGridDims[1]) <= 0) || (floor(tGridDims[2]) <= 0))
        {
            fieldmsg( eError ) << "invalid grid dimensions: " << tGridDims << eom;
            return false;
        }

        fGrid = vtkSmartPointer<vtkImageData>::New();
        fGrid->SetDimensions( (int)floor(tGridDims[0]), (int)floor(tGridDims[1]), (int)floor(tGridDims[2]) );
        fGrid->SetOrigin( tGridOrigin[0], tGridOrigin[1], tGridOrigin[2] );
        fGrid->SetSpacing( fSpacing, fSpacing, fSpacing );

        unsigned int tNumPoints = fGrid->GetNumberOfPoints();
        if (tNumPoints < 1)
        {
            fieldmsg( eError ) << "invalid number of points: " << tNumPoints << eom;
            return false;
        }

        fieldmsg_debug( "grid has "<<tNumPoints<<" points"<<eom );

        int tDims[3];
        fGrid->GetDimensions(tDims);
        fieldmsg_debug("grid dimensions are "
            <<tDims[0]<<"x"<<tDims[1]<<"x"<<tDims[2]
            <<eom);

        double tBounds[6];
        fGrid->GetBounds(tBounds);
        fieldmsg_debug("grid coordinates range from "
            <<"("<<tBounds[0]<<"|"<<tBounds[2]<<"|"<<tBounds[4]<<") to "
            <<"("<<tBounds[1]<<"|"<<tBounds[3]<<"|"<<tBounds[5]<<")"
            <<eom);

        fieldmsg_debug("grid center is "
            <<"("<<0.5*(tBounds[1]+tBounds[0])<<"|"<<0.5*(tBounds[3]+tBounds[2])<<"|"<<0.5*(tBounds[5]+tBounds[4])<<")"
            <<eom);

        if (fMirrorX || fMirrorY || fMirrorZ)
        {
            fieldmsg( eNormal ) <<"mirroring points along "
                <<(fMirrorX ? "x" : "")
                <<(fMirrorY ? "y" : "")
                <<(fMirrorZ ? "z" : "")
                <<"-axis, effective number of points reduced to "
                <<tNumPoints/((fMirrorX?2:1)*(fMirrorY?2:1)*(fMirrorZ?2:1))
                <<eom;
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

        return true;
    }

    bool KSFieldElectricPotentialmapCalculator::Execute()
    {
        fValidityData->FillComponent(0, 0);  // initialize all to 0 = invalid

        unsigned int tNumPoints = fGrid->GetNumberOfPoints();

        //timer
        clock_t tClockStart, tClockEnd;
        double tTimeSpent;


        fieldmsg( eNormal ) << "computing electric potential at " << tNumPoints << " grid points" << eom;

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

            if (! CheckPosition(KThreeVector(tPoint)) )
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
                fElectricField->CalculatePotential(KThreeVector(tPoint), 0, tPotential);
            }

            fPotentialData->SetTuple1(i, tPotential);
            fValidityData->SetTuple1(i, 1);
        }
        std::cout<<std::endl;
        tClockEnd = clock();

        tTimeSpent = ((double)(tClockEnd - tClockStart))/CLOCKS_PER_SEC; // time in seconds
        fieldmsg( eNormal ) << "finished computing potential map (total time spent = " << tTimeSpent << ", time per potential evaluation = " << tTimeSpent/(double)(tNumPoints) << ")" << eom;


        fieldmsg( eNormal ) << "computing electric field at " << tNumPoints << " grid points" << eom;

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

            if (! CheckPosition(KThreeVector(tPoint)) )
                continue;

            bool tHasValue = false;
            KThreeVector tField;

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
                fElectricField->CalculateField(KThreeVector(tPoint), 0, tField);
            }

            fFieldData->SetTuple3(i, tField[0], tField[1], tField[2]);
            fValidityData->SetTuple1(i, 2);
        }
        std::cout<<std::endl;
        tClockEnd = clock();

        tTimeSpent = ((double)(tClockEnd - tClockStart))/CLOCKS_PER_SEC; // time in seconds
        fieldmsg( eNormal ) << "finished computing potential map (total time spent = " << tTimeSpent << ", time per field evaluation = " << tTimeSpent/(double)(tNumPoints) << ")" << eom;

        return true;
    }

    bool KSFieldElectricPotentialmapCalculator::Finish()
    {
        string filename = fDirectory + "/" + fFile;
        fieldmsg( eNormal ) << "exporting vtkImageData file <" << filename << ">" << eom;

        vtkSmartPointer<vtkXMLImageDataWriter> vWriter = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        vWriter->SetFileName(filename.c_str());
#if (VTK_MAJOR_VERSION >= 6)
        vWriter->SetInputData(fGrid);
#else
        vWriter->SetInput(fGrid);
#endif
        vWriter->SetDataModeToBinary();
        vWriter->Write();

        fieldmsg( eNormal ) << "finished writing vtkImageData file" << eom;

        return true;
    }

    void KSFieldElectricPotentialmapCalculator::InitializeComponent()
    {
        if (! this->Prepare())
        {
            fieldmsg( eError ) << "failed to initialize the potential map calculator" << eom;
        }

        if (! this->Execute())
        {
            fieldmsg( eError ) << "failed to execute the potential map calculator" << eom;
        }

        if (! this->Finish())
        {
            fieldmsg( eError ) << "failed to finalize the potential map calculator" << eom;
        }
    }

    void KSFieldElectricPotentialmapCalculator::DeinitializeComponent()
    {
        return;
    }

}
