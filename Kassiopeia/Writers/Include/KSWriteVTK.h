#ifndef _Kassiopeia_KSWriteVTK_h_
#define _Kassiopeia_KSWriteVTK_h_

#include "KSWriter.h"

#include "vtkSmartPointer.h"
#include "vtkVertex.h"
#include "vtkPolyLine.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkXMLPolyDataWriter.h"

#include "vtkCharArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkShortArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkIntArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkLongArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

    class KSWriteVTK :
        public KSComponentTemplate< KSWriteVTK, KSWriter >
    {
        private:
            class Action
            {
                public:
                    Action()
                    {
                    }
                    virtual ~Action()
                    {
                    }

                    virtual void Execute() = 0;
            };

            class UIntAction :
                public Action
            {
                public:
                    UIntAction( unsigned int* aData, vtkSmartPointer< vtkUnsignedIntArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~UIntAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextValue( *fData );
                        return;
                    }

                private:
                    unsigned int* fData;
                    vtkSmartPointer< vtkUnsignedIntArray > fArray;
            };

            class IntAction :
                public Action
            {
                public:
                    IntAction( int* aData, vtkSmartPointer< vtkIntArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~IntAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextValue( *fData );
                        return;
                    }

                private:
                    int* fData;
                    vtkSmartPointer< vtkIntArray > fArray;
            };

            class FloatAction :
                public Action
            {
                public:
                    FloatAction( float* aData, vtkSmartPointer< vtkFloatArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~FloatAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextValue( *fData );
                        return;
                    }

                private:
                    float* fData;
                    vtkSmartPointer< vtkFloatArray > fArray;
            };

            class DoubleAction :
                public Action
            {
                public:
                    DoubleAction( double* aData, vtkSmartPointer< vtkDoubleArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~DoubleAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextValue( *fData );
                        return;
                    }

                private:
                    double* fData;
                    vtkSmartPointer< vtkDoubleArray > fArray;
            };

            class TwoVectorAction :
                public Action
            {
                public:
                    TwoVectorAction( KTwoVector* aData, vtkSmartPointer< vtkDoubleArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~TwoVectorAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextTuple2( fData->X(), fData->Y() );
                        return;
                    }

                private:
                    KTwoVector* fData;
                    vtkSmartPointer< vtkDoubleArray > fArray;
            };

            class ThreeVectorAction :
                public Action
            {
                public:
                    ThreeVectorAction( KThreeVector* aData, vtkSmartPointer< vtkDoubleArray > anArray ) :
                            fData( aData ),
                            fArray( anArray )
                    {
                    }
                    virtual ~ThreeVectorAction()
                    {
                    }

                    void Execute()
                    {
                        fArray->InsertNextTuple3( fData->X(), fData->Y(), fData->Z() );
                        return;
                    }

                private:
                    KThreeVector* fData;
                    vtkSmartPointer< vtkDoubleArray > fArray;
            };

            class PointAction :
                public Action
            {
                public:
                    PointAction( KThreeVector* aData, std::vector< vtkIdType >& anIds, vtkSmartPointer< vtkPoints > aPoints ) :
                            fData( aData ),
                            fIds( anIds ),
                            fPoints( aPoints )
                    {
                    }
                    virtual ~PointAction()
                    {
                    }

                    void Execute()
                    {
                        fIds.push_back( fPoints->InsertNextPoint( fData->X(), fData->Y(), fData->Z() ) );
                        return;
                    }

                private:
                    KThreeVector* fData;
                    std::vector< vtkIdType >& fIds;
                    vtkSmartPointer< vtkPoints > fPoints;

            };

            typedef std::map< KSComponent*, Action* > ActionMap;
            typedef std::pair< KSComponent*, Action* > ActionEntry;
            typedef ActionMap::iterator ActionIt;
            typedef ActionMap::const_iterator ActionCIt;

        public:
            KSWriteVTK();
            KSWriteVTK( const KSWriteVTK& aCopy );
            KSWriteVTK* Clone() const;
            ~KSWriteVTK();

        public:
            void SetBase( const std::string& aBase );
            void SetPath( const std::string& aPath );

        private:
            std::string fBase;
            std::string fPath;

        public:
            void ExecuteRun();
            void ExecuteEvent();
            void ExecuteTrack();
            void ExecuteStep();

            void SetTrackPoint( KSComponent* aComponent );
            void ClearTrackPoint( KSComponent* aComponent );

            void SetTrackData( KSComponent* aComponent );
            void ClearTrackData( KSComponent* aComponent );

            void SetStepPoint( KSComponent* aComponent );
            void ClearStepPoint( KSComponent* aComponent );

            void SetStepData( KSComponent* aComponent );
            void ClearStepData( KSComponent* aComponent );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            void AddTrackPoint( KSComponent* aComponent );
            void AddTrackData( KSComponent* aComponent );
            void FillTrack();
            void BreakTrack();

            bool fTrackPointFlag;
            KSComponent* fTrackPointComponent;
            ActionEntry fTrackPointAction;

            bool fTrackDataFlag;
            KSComponent* fTrackDataComponent;
            ActionMap fTrackDataActions;

            std::vector< vtkIdType > fTrackIds;
            vtkSmartPointer< vtkPoints > fTrackPoints;
            vtkSmartPointer< vtkCellArray > fTrackVertices;
            vtkSmartPointer< vtkPolyData > fTrackData;

            void AddStepPoint( KSComponent* aComponent );
            void AddStepData( KSComponent* aComponent );
            void FillStep();
            void BreakStep();

            bool fStepPointFlag;
            KSComponent* fStepPointComponent;
            ActionEntry fStepPointAction;

            bool fStepDataFlag;
            KSComponent* fStepDataComponent;
            ActionMap fStepDataActions;

            std::vector< vtkIdType > fStepIds;
            vtkSmartPointer< vtkPoints > fStepPoints;
            vtkSmartPointer< vtkCellArray > fStepLines;
            vtkSmartPointer< vtkPolyData > fStepData;
    };

}

#endif
