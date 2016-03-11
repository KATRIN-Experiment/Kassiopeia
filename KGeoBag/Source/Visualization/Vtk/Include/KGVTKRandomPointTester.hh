#ifndef KGVTKNEARESTNORMALPAINTER_HH_
#define KGVTKNEARESTNORMALPAINTER_HH_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KGCore.hh"
#include "KGRGBColor.hh"

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"

#include "KField.h"

namespace KGeoBag
{

    class KGVTKRandomPointTester :
        public KVTKPainter
    {
        public:
    		KGVTKRandomPointTester();
            virtual ~KGVTKRandomPointTester();

            void Render();
            void Display();
            void Write();

            void AddSurface( const KGSurface* aSurface );
            void AddSpace( const KGSpace* aSpace );

            K_SET( KGRGBColor, SampleColor )
            K_SET( double, VertexSize )
            K_SET( std::vector<KThreeVector*>, SamplePoints )

        private:
            vector< const KGSurface* > fSurfaces;
            vector< const KGSpace* > fSpaces;

            vtkSmartPointer< vtkPoints > fPoints;
            vtkSmartPointer< vtkUnsignedCharArray > fColors;
            vtkSmartPointer< vtkCellArray > fCells;
            vtkSmartPointer< vtkPolyData > fPolyData;
            vtkSmartPointer< vtkPolyDataMapper > fMapper;
            vtkSmartPointer< vtkActor > fActor;
    };


    inline void KGVTKRandomPointTester::AddSurface( const KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }

    inline void KGVTKRandomPointTester::AddSpace( const KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

}

#endif
