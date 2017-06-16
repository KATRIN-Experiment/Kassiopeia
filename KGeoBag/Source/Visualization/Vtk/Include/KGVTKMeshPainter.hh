#ifndef KGVTKMESHPAINTER_HH_
#define KGVTKMESHPAINTER_HH_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KGCore.hh"
#include "KGMesh.hh"

#include "vtkSmartPointer.h"
#include "vtkLookupTable.h"
#include "vtkDoubleArray.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"

namespace KGeoBag
{

    class KGVTKMeshPainter :
        public KVTKPainter,
        public KGVisitor,
        public KGSurface::Visitor,
        public KGExtendedSurface< KGMesh >::Visitor,
        public KGSpace::Visitor,
        public KGExtendedSpace< KGMesh >::Visitor
    {
        public:
            KGVTKMeshPainter();
            virtual ~KGVTKMeshPainter();

        public:
            void Render();
            void Display();
            void Write();

            void VisitSurface( KGSurface* aSurface );
            void VisitExtendedSurface( KGExtendedSurface< KGMesh >* aSurface );
            void VisitSpace( KGSpace* aSpace );
            void VisitExtendedSpace( KGExtendedSpace< KGMesh >* aSpace );

            void SetFile( const std::string& aName );
            const std::string& GetFile() const;

            void SetArcCount( const unsigned int& anArcCount );
            const unsigned int& GetArcCount() const;

            void SetColorMode( const unsigned int& aColorMode );
            const unsigned int& GetColorMode() const;

            static const unsigned int sArea;
            static const unsigned int sAspect;
            static const unsigned int sModulo;

        private:
            void PaintElements();

            KThreeVector fCurrentOrigin;
            KThreeVector fCurrentXAxis;
            KThreeVector fCurrentYAxis;
            KThreeVector fCurrentZAxis;

            KGMeshElementVector* fCurrentElements;

            vtkSmartPointer< vtkLookupTable > fColorTable;
            vtkSmartPointer< vtkDoubleArray > fAreaData;
            vtkSmartPointer< vtkDoubleArray > fAspectData;
            vtkSmartPointer< vtkDoubleArray > fModuloData;
            vtkSmartPointer< vtkPoints > fPoints;
            vtkSmartPointer< vtkCellArray > fLineCells;
            vtkSmartPointer< vtkCellArray > fPolyCells;
            vtkSmartPointer< vtkPolyData > fPolyData;
            vtkSmartPointer< vtkPolyDataMapper > fMapper;
            vtkSmartPointer< vtkActor > fActor;

            std::string fFile;
            unsigned int fArcCount;
            unsigned int fColorMode;
    };

}

#endif
