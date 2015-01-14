#include "KROOTPad.h"
#include "KROOTPainter.h"
#include "KUtilityMessage.h"
#include "TStyle.h"

#include <math.h>
#include <limits>

namespace katrin
{

	KROOTPad::KROOTPad() :
			fPainters(),
			fFrame( 0 ),
			fPad( 0 ),
			fWindow( 0 ),
			fxlow( 0. ),
			fylow( 0. ),
			fxup( 1.0 ),
			fyup( 1.0 )
    {
    }

	KROOTPad::~KROOTPad()
    {
        return;
    }

    void KROOTPad::Render()
    {
    	utilmsg( eNormal ) <<"KROOTPad starts to render!"<<eom;

        gStyle->SetPadBottomMargin(0.1);
      	gStyle->SetPadRightMargin(0.05);
       	gStyle->SetPadLeftMargin(0.05);
       	gStyle->SetPadTopMargin(0.05);
        gStyle->SetTitleX( 0.5 );
        gStyle->SetTitleAlign( 23 );
        gStyle->SetTitleSize( 0.08 , "t" );

        fPad = new TPad( fName.c_str() , fName.c_str(), fxlow, fylow, fxup, fyup );

        double tXMin(std::numeric_limits<double>::max());
        double tXMax(-1.0*std::numeric_limits<double>::max());
        double tYMin(std::numeric_limits<double>::max());
        double tYMax(-1.0*std::numeric_limits<double>::max());

        /* render painters */
        PainterIt tIt;
        if ( fPainters.size() > 0 )
        {
			for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
			{
				(*tIt)->Render();
				double tLocalXMin = (*tIt)->GetXMin();
				if ( tLocalXMin < tXMin ) tXMin = tLocalXMin;
				double tLocalXMax = (*tIt)->GetXMax();
				if ( tLocalXMax > tXMax ) tXMax = tLocalXMax;
				double tLocalYMin = (*tIt)->GetYMin();
				if ( tLocalYMin < tYMin ) tYMin = tLocalYMin;
				double tLocalYMax = (*tIt)->GetYMax();
				if ( tLocalYMax > tYMax ) tYMax = tLocalYMax;
			}

			utilmsg_debug( "XMin: "<<tXMin<<eom);
			utilmsg_debug( "XMax: "<<tXMax<<eom);
			utilmsg_debug( "YMin: "<<tYMin<<eom);
			utilmsg_debug( "YMax: "<<tYMax<<eom);

			tXMin = tXMin - ( tXMax - tXMin )/20.;
			tXMax = tXMax + ( tXMax - tXMin )/20.;
			tYMin = tYMin - ( tYMax - tYMin )/20.;
			tYMax = tYMax + ( tYMax - tYMin )/20.;

			if ( tXMin == tXMax)
			{
				tXMin = tXMin - tXMin / 20.;
				tXMax = tXMax + tXMax / 20.;
			}

			if ( tYMin == tYMax)
			{
				tYMin = tYMin - tYMin / 20.;
				tYMax = tYMax + tYMax / 20.;
			}

			Int_t tNBins = 1000;
			fFrame = new TH2F( GetName().c_str(), "", tNBins, tXMin, tXMax, tNBins, tYMin, tYMax);
			fFrame->SetStats(0);

			//take axis label of last painter
			KROOTPainter* tLastPainter = fPainters.at( fPainters.size() - 1 );
			if ( tLastPainter )
			{
				fFrame->GetXaxis()->SetTitle( tLastPainter->GetXAxisLabel().c_str() );
				fFrame->GetYaxis()->SetTitle( tLastPainter->GetYAxisLabel().c_str() );
			}
			fFrame->GetXaxis()->SetTitleSize( 0.04 );
			fFrame->GetXaxis()->SetTitleOffset( 1.0 );
			fFrame->GetXaxis()->SetLabelSize( 0.05 );
			fFrame->GetYaxis()->SetTitleSize( 0.04 );
			fFrame->GetYaxis()->SetTitleOffset( 0.4 );
			fFrame->GetYaxis()->SetLabelSize( 0.05 );

        }

        utilmsg( eNormal ) <<"KROOTPad finished to render!"<<eom;

        return;
    }

    void KROOTPad::Display()
    {
    	utilmsg( eNormal ) <<"KROOTPad starts to display!"<<eom;

    	fPad->Draw();
    	fPad->cd();

    	if ( fFrame )
    	{
			fFrame->Draw( "axis" );
    	}

        /* display painters */
        PainterIt tIt;
        for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
        {
            (*tIt)->Display();
        }

    	utilmsg( eNormal ) <<"KROOTPad finished to display!"<<eom;
        return;
    }

    void KROOTPad::Write()
    {
        return;
    }

    void KROOTPad::AddPainter( KPainter* aPainter )
    {
    	KROOTPainter* tPainter = dynamic_cast< KROOTPainter* >( aPainter );
        if( tPainter != NULL )
        {
            fPainters.push_back( tPainter );
			tPainter->SetWindow( this );
			return;
        }
        utilmsg( eError ) << "cannot add non-root painter <" << aPainter->GetName() << "> to root pad <" << GetName() << ">" << eom;
        return;
    }
    void KROOTPad::RemovePainter( KPainter* aPainter )
    {
    	KROOTPainter* tPainter = dynamic_cast< KROOTPainter* >( aPainter );
        if( tPainter != NULL )
        {
            PainterIt tIt;
            for( tIt = fPainters.begin(); tIt != fPainters.end(); tIt++ )
            {
                if ( (*tIt) == tPainter )
                {
                	fPainters.erase( tIt );
                	tPainter->ClearWindow( this );
                	return;
                }
            }
            utilmsg( eError ) << "cannot remove root painter <" << tPainter->GetName() << "> from root pad <" << GetName() << ">" << eom;
        }
        utilmsg( eError ) << "cannot remove non-root painter <" << aPainter->GetName() << "> from root pad <" << GetName() << ">" << eom;
        return;
    }

    void KROOTPad::SetWindow( KWindow* aWindow )
    {
    	KROOTWindow* tWindow = dynamic_cast< KROOTWindow* >( aWindow );
        if( tWindow != NULL )
        {
            if( fWindow == NULL )
            {
                fWindow = tWindow;
                return;
            }
            utilmsg( eError ) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        }
        utilmsg( eError ) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        return;
    }

    void KROOTPad::ClearWindow( KWindow* aWindow )
    {
    	KROOTWindow* tWindow = dynamic_cast< KROOTWindow* >( aWindow );
        if( tWindow != NULL )
        {
            if( fWindow == tWindow )
            {
                fWindow = NULL;
                return;
            }
            utilmsg( eError ) << "cannot use root window <" << tWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
        }
        return;
        utilmsg( eError ) << "cannot use non-root window <" << aWindow->GetName() << "> with root painter <" << GetName() << ">" << eom;
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
#include "KROOTWindow.h"

namespace katrin
{

    static int sKROOTPadStructure =
        KROOTPadBuilder::Attribute< string >( "name" ) +
        KROOTPadBuilder::Attribute< double >( "xlow" ) +
        KROOTPadBuilder::Attribute< double >( "ylow" ) +
        KROOTPadBuilder::Attribute< double >( "xup" ) +
        KROOTPadBuilder::Attribute< double >( "yup" );


}
