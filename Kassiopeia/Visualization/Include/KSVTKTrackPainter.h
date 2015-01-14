#ifndef _Kassiopeia_KSVTKTrackPainter_h_
#define _Kassiopeia_KSVTKTrackPainter_h_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "vtkSmartPointer.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkLookupTable.h"
#include "vtkActor.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSVTKTrackPainter :
        public KVTKPainter
    {
        public:
            KSVTKTrackPainter();
            ~KSVTKTrackPainter();

            void Render();
            void Display();
            void Write();

            ;K_SET( string, File )
            ;K_SET( string, Path )
            ;K_SET( string, OutFile )
            ;K_SET( string, PointObject )
            ;K_SET( string, PointVariable )
            ;K_SET( string, ColorObject )
            ;K_SET( string, ColorVariable )

        private:
            vtkSmartPointer< vtkPoints > fPoints;
            vtkSmartPointer< vtkCellArray > fLines;
            vtkSmartPointer< vtkDoubleArray > fColors;
            vtkSmartPointer< vtkPolyData > fData;
            vtkSmartPointer< vtkPolyDataMapper > fMapper;
            vtkSmartPointer< vtkLookupTable > fTable;
            vtkSmartPointer< vtkActor > fActor;
    };

}

#endif
