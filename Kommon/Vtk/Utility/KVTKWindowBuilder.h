//
// Created by trost on 25.07.16.
//

#ifndef KASPER_KVTKWINDOWBUILDER_H
#define KASPER_KVTKWINDOWBUILDER_H

#include "KComplexElement.hh"
#include "KVTKWindow.h"
#include "KVTKPainter.h"

namespace katrin
{

typedef KComplexElement< KVTKWindow > KVTKWindowBuilder;

template< >
inline bool KVTKWindowBuilder::Begin()
{
    fObject = new KVTKWindow();
    return true;
}

template< >
inline bool KVTKWindowBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        aContainer->CopyTo( fObject, &KNamed::SetName );
        return true;
    }

    if( aContainer->GetName() == "enable_write" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetWriteMode );
        return true;
    }
    if( aContainer->GetName() == "enable_display" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetDisplayMode );
        return true;
    }
    if( aContainer->GetName() == "enable_help" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetHelpMode );
        return true;
    }
    if( aContainer->GetName() == "enable_axis" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetAxisMode );
        return true;
    }
    if( aContainer->GetName() == "enable_data" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetDataMode );
        return true;
    }
    if( aContainer->GetName() == "enable_parallel_projection" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetParallelProjectionMode );
        return true;
    }

    if( aContainer->GetName() == "frame_title" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameTitle );
        return true;
    }
    if( aContainer->GetName() == "frame_size_x" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameSizeX );
        return true;
    }
    if( aContainer->GetName() == "frame_size_y" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameSizeY );
        return true;
    }
    if( aContainer->GetName() == "frame_color_red" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameColorRed );
        return true;
    }
    if( aContainer->GetName() == "frame_color_green" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameColorGreen );
        return true;
    }
    if( aContainer->GetName() == "frame_color_blue" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetFrameColorBlue );
        return true;
    }
    if( aContainer->GetName() == "view_angle" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetViewAngle );
        return true;
    }
    if( aContainer->GetName() == "eye_angle" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetEyeAngle );
        return true;
    }

    if( aContainer->GetName() == "multi_samples" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetMultiSamples );
        return true;
    }
    if( aContainer->GetName() == "depth_peeling" )
    {
        aContainer->CopyTo( fObject, &KVTKWindow::SetDepthPeelingLevel );
        return true;
    }
    return false;
}

template< >
inline bool KVTKWindowBuilder::AddElement( KContainer* aContainer )
{
    if( aContainer->Is< KVTKPainter >() == true )
    {
        aContainer->ReleaseTo( fObject, &KVTKWindow::AddPainter );
        return true;
    }
    return false;
}

template< >
inline bool KVTKWindowBuilder::End()
{
    fObject->Render();
    fObject->Write();
    fObject->Display();
    delete fObject;
    return true;
}

}

#endif //KASPER_KVTKWINDOWBUILDER_H
