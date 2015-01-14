#include "KVTKWindow.h"
#include "KVTKPainter.h"
#include "KUtilityMessage.h"

#include "vtkSmartPointer.h"
#include "vtkAxesActor.h"
#include "vtkAnnotatedCubeActor.h"
#include "vtkAnnotatedCubeActor.h"
#include "vtkAppendPolyData.h"
#include "vtkCallbackCommand.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkLightKit.h"
#include "vtkLODActor.h"
#include "vtkMapper.h"
#include "vtkPNGWriter.h"
#include "vtkTIFFWriter.h"
#include "vtkJPEGWriter.h"
#include "vtkPointData.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataMapper2D.h"
#include "vtkPropCollection.h"
#include "vtkProperty.h"
#include "vtkQuad.h"
#include "vtkTextProperty.h"
#include "vtkTriangleStrip.h"
#include "vtkWindowToImageFilter.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkCaptionActor2D.h"

#include <iostream>
using std::cout;
using std::endl;

namespace katrin
{

    KVTKWindow::KVTKWindow() :
            fWriteToggle( true ),
            fDisplayToggle( true ),
            fHelpToggle( false ),
            fDataToggle( true ),
            fAxisToggle( true ),
            fParallelProjectionToggle( false ),
            fFrameTitle( "katrin Visualization" ),
            fFrameXPixels( 800 ),
            fFrameYPixels( 600 ),
            fFrameRed( 0.2 ),
            fFrameGreen( 0.2 ),
            fFrameBlue( 0.2 ),
            fEyeAngle( 0.5 ),
            fViewAngle( 45. ),
            fMultiSamples( 8 ),
            fDepthPeelingLevel( 10 )
    {
    }
    KVTKWindow::~KVTKWindow()
    {
        return;
    }

    void KVTKWindow::Render()
    {
        /* setup writer */
        if( fWriteToggle == true )
        {
            fWriter = vtkSmartPointer< vtkXMLPolyDataWriter >::New();
        }

        /* setup display */
        if( fDisplayToggle == true )
        {
            double textColor[] =
            { fFrameRed < .5 ? 1. : 0, fFrameGreen < .5 ? 1. : 0, fFrameBlue < .5 ? 1. : 0 };

            /* setup renderer */
            fRenderer = vtkSmartPointer< vtkRenderer >::New();
            fRenderer->SetBackground( fFrameRed, fFrameGreen, fFrameBlue );
            fRenderer->GetActiveCamera()->SetClippingRange( .00025, 250 );
            fRenderer->GetActiveCamera()->SetViewAngle( fViewAngle );
            fRenderer->GetActiveCamera()->SetEyeAngle( fEyeAngle );
            fRenderer->RemoveAllLights();

            /* setup light kit, consisting of 5 lights overall */
            vtkSmartPointer< vtkLightKit > lightKit = vtkSmartPointer< vtkLightKit >::New();
            lightKit->SetKeyLightIntensity( 1. );
            lightKit->SetKeyToFillRatio( 4 );
            lightKit->SetKeyToHeadRatio( 12 );
            lightKit->SetKeyToBackRatio( 6 );
            lightKit->AddLightsToRenderer( fRenderer );

            /* setup render window, with 8x FSAA enabled */
            fRenderWindow = vtkSmartPointer< vtkRenderWindow >::New();
            fRenderWindow->SetSize( fFrameXPixels, fFrameYPixels );
            fRenderWindow->SetWindowName( fFrameTitle.c_str() );
            fRenderWindow->SwapBuffersOn();
            fRenderWindow->StereoCapableWindowOn();
            fRenderWindow->SetStereoType( VTK_STEREO_ANAGLYPH );
            fRenderWindow->SetAnaglyphColorSaturation( 0.50 ); // default is 0.65
            fRenderWindow->SetDesiredUpdateRate( 30 );
            fRenderWindow->AddRenderer( fRenderer );

            /* setup depth peeling or multisamples */
            if( fDepthPeelingLevel > 0 )
            {
                fRenderWindow->SetAlphaBitPlanes( 1 );
                fRenderWindow->SetMultiSamples( 0 );
                fRenderer->SetUseDepthPeeling( 1 );
                fRenderer->SetMaximumNumberOfPeels( fDepthPeelingLevel );
                fRenderer->SetOcclusionRatio( 0.05 );
            }
            else
            {
                fRenderWindow->SetMultiSamples( fMultiSamples );
                fRenderer->SetUseDepthPeeling( 0 );
            }

            /* setup window interactor, use trackball-like camera */
            vtkSmartPointer< vtkInteractorStyleTrackballCamera > tInteractorStyle = vtkSmartPointer< vtkInteractorStyleTrackballCamera >::New();
            tInteractorStyle->SetAutoAdjustCameraClippingRange( 1 );
            tInteractorStyle->HandleObserversOff();

            fRenderInteractor = vtkSmartPointer< vtkRenderWindowInteractor >::New();
            fRenderInteractor->SetInteractorStyle( tInteractorStyle );
            fRenderInteractor->SetRenderWindow( fRenderWindow );

            /* setup help actor */
            fHelpActor = vtkSmartPointer< vtkCornerAnnotation >::New();
            fHelpActor->GetTextProperty()->SetFontFamilyToCourier();
            fHelpActor->GetTextProperty()->SetBold( 1 );
            fHelpActor->GetTextProperty()->SetFontSize( 8 );
            fHelpActor->GetTextProperty()->SetColor( textColor );
            fHelpActor->GetTextProperty()->SetLineSpacing( 0.8 );
            fHelpActor->GetTextProperty()->SetJustificationToLeft();
            UpdateHelp();

            /* setup data actor */
            fDataActor = vtkSmartPointer< vtkCornerAnnotation >::New();
            fDataActor->GetTextProperty()->SetFontFamilyToCourier();
            fDataActor->GetTextProperty()->SetBold( 1 );
            fDataActor->GetTextProperty()->SetFontSize( 8 );
            fDataActor->GetTextProperty()->SetColor( textColor );
            fDataActor->GetTextProperty()->SetLineSpacing( 0.8 );
            fDataActor->GetTextProperty()->SetJustificationToRight();
            UpdateData();

            /* setup axes actor */
            fAxesActor = vtkSmartPointer< vtkAxesActor >::New();
            fAxesActor->SetConeRadius( .3 );
            fAxesActor->SetScale( 1.0 );
            fAxesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOff();
            fAxesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOff();
            fAxesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOff();
            fAxesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor( textColor );
            fAxesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor( textColor );
            fAxesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor( textColor );

            fOrientationWidget = vtkSmartPointer< vtkOrientationMarkerWidget >::New();
            fOrientationWidget->SetOrientationMarker( fAxesActor );
            fOrientationWidget->SetViewport( 0.0, 0.0, 0.2, 0.2 );
            fOrientationWidget->SetInteractor( fRenderInteractor );
            fOrientationWidget->EnabledOn();
            fOrientationWidget->InteractiveOff();
            fOrientationWidget->EnabledOff();

            /* setup callback commands */
            vtkSmartPointer< vtkCallbackCommand > tKeyPressEvent = vtkSmartPointer< vtkCallbackCommand >::New();
            tKeyPressEvent->SetCallback( &KVTKWindow::OnKeyPress );
            tKeyPressEvent->SetClientData( this );
            fRenderInteractor->AddObserver( vtkCommand::KeyPressEvent, tKeyPressEvent );

            vtkSmartPointer< vtkCallbackCommand > tEndEvent = vtkSmartPointer< vtkCallbackCommand >::New();
            tEndEvent->SetCallback( &KVTKWindow::OnEnd );
            tEndEvent->SetClientData( this );
            fRenderer->AddObserver( vtkCommand::EndEvent, tEndEvent );

            /* initialize interactor */
            fRenderInteractor->Initialize();
        }

        /* render painters */
        PainterIt tIt;
        for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
        {
            (*tIt)->Render();
        }

        return;
    }

    void KVTKWindow::Display()
    {
        PainterIt tIt;
        if( fDisplayToggle == true )
        {
            /* display painters */
            for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
            {
                (*tIt)->Display();
            }

            /* add help actor if necessary */
            if( fHelpToggle == true )
            {
                fRenderer->AddActor( fHelpActor );
            }

            /* add data actor if necessary */
            if( fDataToggle == true )
            {
                fRenderer->AddActor( fDataActor );
            }

            /* add axis actor if necessary */
            if( fAxisToggle == true )
            {
                fOrientationWidget->EnabledOn();
            }

            /* enable parallel projection if necessary */
            if( fParallelProjectionToggle == true )
            {
                fRenderer->GetActiveCamera()->SetParallelProjection( 1 );
            }

            /* setup renderer */
            fRenderer->ResetCamera();
            fRenderWindow->Render();
            fRenderInteractor->Start();
        }

        return;
    }

    void KVTKWindow::Write()
    {
        PainterIt tIt;
        if( fWriteToggle == true )
        {
            /* write painters */
            for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
            {
                (*tIt)->Write();
            }
        }

        return;
    }

    void KVTKWindow::AddPainter( KPainter* aPainter )
    {
        KVTKPainter* tPainter = dynamic_cast< KVTKPainter* >( aPainter );
        if( tPainter != NULL )
        {
            if( fPainters.insert( tPainter ).second == true )
            {
                tPainter->SetWindow( this );
                return;
            }
            utilmsg( eError ) << "cannot add vtk painter <" << tPainter->GetName() << "> to vtk window <" << GetName() << ">" << eom;
        }
        utilmsg( eError ) << "cannot add non-vtk painter <" << aPainter->GetName() << "> to vtk window <" << GetName() << ">" << eom;
        return;
    }
    void KVTKWindow::RemovePainter( KPainter* aPainter )
    {
        KVTKPainter* tPainter = dynamic_cast< KVTKPainter* >( aPainter );
        if( tPainter != NULL )
        {
            if( fPainters.erase( tPainter ) == 1 )
            {
                tPainter->ClearWindow( this );
                return;
            }
            utilmsg( eError ) << "cannot remove vtk painter <" << tPainter->GetName() << "> from vtk window <" << GetName() << ">" << eom;
        }
        utilmsg( eError ) << "cannot remove non-vtk painter <" << aPainter->GetName() << "> from vtk window <" << GetName() << ">" << eom;
        return;
    }

    void KVTKWindow::AddActor( vtkSmartPointer< vtkActor > anActor )
    {
        fActors.push_back( anActor );
        return;
    }
    void KVTKWindow::RemoveActor( vtkSmartPointer< vtkActor > anActor )
    {
        ActorIt tIter;
        for( tIter = fActors.begin(); tIter != fActors.end(); tIter++ )
        {
            if( anActor == *tIter )
            {
                fActors.erase( tIter );
                return;
            }
        }
        return;
    }

    void KVTKWindow::AddPoly( vtkSmartPointer< vtkPolyData > aPoly )
    {
        fPolys.push_back( aPoly );
        return;
    }
    void KVTKWindow::RemovePoly( vtkSmartPointer< vtkPolyData > aPoly )
    {
        PolyIt tIter;
        for( tIter = fPolys.begin(); tIter != fPolys.end(); tIter++ )
        {
            if( aPoly == *tIter )
            {
                fPolys.erase( tIter );
                return;
            }
        }
        return;
    }

    void KVTKWindow::UpdateHelp()
    {
        std::ostringstream tText;
        tText << std::setiosflags( std::ios_base::right | std::ios_base::fixed );

        tText << "mouse interaction:" << '\n';
        tText << "  rotation - left button [all mice]" << '\n';
        tText << "  pan - center button [3 button mouse], shift + left button [1 or 2 button mouse]" << '\n';
        tText << "  zoom - right button [3 button mouse], ctrl + shift + left button [1 or 2 button mouse]" << '\n';
        tText << '\n';
        tText << "help toggle: h                    [" << (fHelpToggle ? "ON" : "OFF") << "]" << '\n';
        tText << "data toggle: d                    [" << (fDataToggle ? "ON" : "OFF") << "]" << '\n';
        tText << "axis toggle: a                    [" << (fAxisToggle ? "ON" : "OFF") << "]" << '\n';
        tText << "parallel projection toggle: p     [" << (fParallelProjectionToggle ? "ON" : "OFF") << "]" << '\n';
        tText << '\n';
        tText << "take screenshot: s" << '\n';
        tText << "reset view: r" << '\n';
        tText << "quit: q" << '\n';

        fHelpActor->SetText( 2, tText.str().c_str() );

        return;
    }
    void KVTKWindow::UpdateData()
    {
        double cameraPosition[ 3 ], cameraFocus[ 3 ];
        fRenderer->GetActiveCamera()->GetPosition( cameraPosition );
        fRenderer->GetActiveCamera()->GetFocalPoint( cameraFocus );

        std::ostringstream tText;
        tText << std::setiosflags( std::ios_base::right | std::ios_base::fixed );

        tText << "position:    [ " << std::setprecision( 3 ) << cameraPosition[ 0 ] << ", " << cameraPosition[ 1 ] << ", " << cameraPosition[ 2 ] << " ]" << '\n';
        tText << "view point:  [ " << std::setprecision( 3 ) << cameraFocus[ 0 ] << ", " << cameraFocus[ 1 ] << ", " << cameraFocus[ 2 ] << " ]" << '\n';

        fDataActor->SetText( 1, tText.str().c_str() );
    }

    void KVTKWindow::Screenshot()
    {
        /* get timestamp for screenshot file */
        time_t tTimeRaw;
        time( &tTimeRaw );

        struct tm* tTimeInfo;
        tTimeInfo = localtime( &tTimeRaw );

        char tTimeStamp[ 64 ];
        strftime( tTimeStamp, 64, "Screenshot_%Y_%m_%d_%H:%M:%S", tTimeInfo );

        /* take screenshot, make sure all actors are rendered at full detail */
        int rate = fRenderWindow->GetDesiredUpdateRate();
        fRenderWindow->SetDesiredUpdateRate( 0 );
        fRenderer->Render();

        vtkSmartPointer< vtkWindowToImageFilter > filter = vtkSmartPointer< vtkWindowToImageFilter >::New();
        filter->SetInput( fRenderWindow );
        filter->Update();

        fRenderWindow->SetDesiredUpdateRate( rate );

        string filename = string( SCRATCH_DEFAULT_DIR ) + string( "/" ) + string( tTimeStamp ) + ".png";
        vtkSmartPointer< vtkPNGWriter > writer = vtkSmartPointer< vtkPNGWriter >::New();
        writer->SetFileName( filename.c_str() );
#ifdef VTK6
        writer->SetInputData( filter->GetOutput() );
#else
        writer->SetInput( filter->GetOutput() );
#endif
        writer->Write();
        utilmsg( eNormal ) << "screenshot saved to <" << filename << ">" << eom;

        return;
    }

    void KVTKWindow::OnKeyPress( vtkObject* aCaller, long unsigned int /*eventId*/, void* aClient, void* /*callData*/)
    {
        KVTKWindow* tWindow = static_cast< KVTKWindow* >( aClient );
        vtkRenderWindowInteractor* tInteractor = static_cast< vtkRenderWindowInteractor* >( aCaller );

        string Symbol = tInteractor->GetKeySym();
        bool WithShift = tInteractor->GetShiftKey();
        bool WithCtrl = tInteractor->GetControlKey();

        if( (WithShift == false) && (WithCtrl == false) )
        {
            //screenshot
            if( Symbol == string( "s" ) )
            {
                tWindow->Screenshot();
            }

            //reset
            else if( Symbol == string( "r" ) )
            {
                tWindow->fRenderer->ResetCamera();
                tWindow->fViewAngle = 45.;
                tWindow->fRenderer->GetActiveCamera()->SetViewAngle( tWindow->fViewAngle );
            }

            //help toggle
            else if( Symbol == string( "h" ) )
            {
                if( tWindow->fHelpToggle == false )
                {
                    tWindow->fRenderer->AddActor( tWindow->fHelpActor );
                    tWindow->fHelpToggle = true;
                }
                else
                {
                    tWindow->fRenderer->RemoveActor( tWindow->fHelpActor );
                    tWindow->fHelpToggle = false;
                }
            }

            //data toggle
            else if( Symbol == string( "d" ) )
            {
                if( tWindow->fDataToggle == false )
                {
                    tWindow->fRenderer->AddActor( tWindow->fDataActor );
                    tWindow->fDataToggle = true;
                }
                else
                {
                    tWindow->fRenderer->RemoveActor( tWindow->fDataActor );
                    tWindow->fDataToggle = false;
                }
            }

            //axis toggle
            else if( Symbol == string( "a" ) )
            {
                if( tWindow->fAxisToggle == false )
                {
                    tWindow->fOrientationWidget->EnabledOn();
                    tWindow->fAxisToggle = true;
                }
                else
                {
                    tWindow->fOrientationWidget->EnabledOff();
                    tWindow->fAxisToggle = false;
                }
            }

            //parallel projection toggle
            else if( Symbol == string( "p" ) )
            {
                if( tWindow->fParallelProjectionToggle == false )
                {
                    tWindow->fRenderer->GetActiveCamera()->SetParallelProjection( 1 );
                    tWindow->fParallelProjectionToggle = true;
                }
                else
                {
                    tWindow->fRenderer->GetActiveCamera()->SetParallelProjection( 0 );
                    tWindow->fParallelProjectionToggle = false;
                }
            }
        }

        tWindow->UpdateHelp();
        tWindow->UpdateData();
        tWindow->fRenderWindow->Render();

        return;
    }

    void KVTKWindow::OnEnd( vtkObject* /*aCaller*/, long unsigned int /*eventId*/, void* aClient, void* /*callData*/)
    {
        KVTKWindow* tWindow = static_cast< KVTKWindow* >( aClient );

        tWindow->UpdateData();

        return;
    }

}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KElementProcessor.hh"

namespace katrin
{

    static int sKVTKWindow =
        KElementProcessor::ComplexElement< KVTKWindow >( "vtk_window" );

    static int sKVTKWindowStructure =
        KVTKWindowBuilder::Attribute< string >( "name" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_display" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_write" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_help" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_axis" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_data" ) +
        KVTKWindowBuilder::Attribute< bool >( "enable_parallel_projection" ) +
        KVTKWindowBuilder::Attribute< string >( "frame_title" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "frame_size_x" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "frame_size_y" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_red" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_green" ) +
        KVTKWindowBuilder::Attribute< float >( "frame_color_blue" ) +
        KVTKWindowBuilder::Attribute< double >( "eye_angle" ) +
        KVTKWindowBuilder::Attribute< double >( "view_angle" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "multi_samples" ) +
        KVTKWindowBuilder::Attribute< unsigned int >( "depth_peeling" );

}
