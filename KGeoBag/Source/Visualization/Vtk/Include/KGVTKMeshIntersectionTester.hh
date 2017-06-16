#ifndef KGVTKNEARESTNORMALPAINTER_HH_
#define KGVTKNEARESTNORMALPAINTER_HH_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KGCore.hh"
#include "KGCore.hh"
#include "KGMesh.hh"
#include "KGRGBColor.hh"

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"

#include "KField.h"

#include "KGNavigableMeshTree.hh"
#include "KGNavigableMeshTreeBuilder.hh"
#include "KGMeshElementCollector.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshFirstIntersectionFinder.hh"
#include "KGNavigableMeshIntersectionFinder.hh"
#include "KGNavigableMeshTreeInformationExtractor.hh"
#include "KGCube.hh"
#include "KGBall.hh"

namespace KGeoBag
{

    class KGVTKMeshIntersectionTester :
        public KVTKPainter
    {
        public:
            KGVTKMeshIntersectionTester();
            virtual ~KGVTKMeshIntersectionTester();

            void Render();
            void Display();
            void Write();

            void AddSurface(KGSurface* aSurface );
            void AddSpace(KGSpace* aSpace );

            void SetSampleCount(unsigned int s){fSampleCount = s;};
            void SetSampleColor(KGRGBColor c){fSampleColor = c;};
            void SetPointColor(KGRGBColor c){fPointColor = c;};
            void SetUnintersectedLineColor(KGRGBColor c){fUnintersectedLineColor = c;};
            void SetIntersectedLineColor(KGRGBColor c){fIntersectedLineColor = c;};
            void SetVertexSize(double s){fVertexSize = s;};
            void SetLineSize(double s){fLineSize = s;};

        private:

            void Construct();

            //mesh container and tree
            KGNavigableMeshElementContainer fContainer;
            KGNavigableMeshTree fTree;
            //KGNavigableMeshFirstIntersectionFinder fFirstIntersectionCalculator;
            KGNavigableMeshFirstIntersectionFinder fIntersectionCalculator;
            KGBall<KGMESH_DIM> fBoundingBall;


            unsigned int fSampleCount;
            KGRGBColor fSampleColor;
            KGRGBColor fPointColor;
            KGRGBColor fUnintersectedLineColor;
            KGRGBColor fIntersectedLineColor;
            double fVertexSize;
            double fLineSize;

            //surface arrays
            std::vector< KGSurface* > fSurfaces;
            std::vector< KGSpace* > fSpaces;

            //vtk data
            vtkSmartPointer< vtkPoints > fPoints;
            vtkSmartPointer< vtkUnsignedCharArray > fColors;
            vtkSmartPointer< vtkCellArray > fPointCells;
            vtkSmartPointer< vtkCellArray > fLineCells;
            vtkSmartPointer< vtkPolyData > fPolyData;
            vtkSmartPointer< vtkPolyDataMapper > fMapper;
            vtkSmartPointer< vtkActor > fActor;
    };

    inline void KGVTKMeshIntersectionTester::AddSurface(KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }

    inline void KGVTKMeshIntersectionTester::AddSpace(KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }


}

#endif
