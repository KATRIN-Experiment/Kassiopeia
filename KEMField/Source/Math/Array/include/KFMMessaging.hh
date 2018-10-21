#ifndef KFMMessaging_HH__
#define KFMMessaging_HH__

#define KFM_USE_KEMCOUT

#ifdef KFM_USE_KEMCOUT
#include "KComplexStreamer.hh"
#include "KDataDisplay.hh"
#include "KEMCout.hh"
#endif

#include <cstdlib>
#include <iostream>

/*
*
*@file KFMMessaging.hh
*@class KFMMessaging
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov  6 12:35:03 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#ifdef KFM_USE_KEMCOUT
#define kfmout KEMField::cout
#define kfmendl KEMField::endl
#define kfmexit std::exit
#else
#define kfmout std::cout
#define kfmendl std::endl
#define kfmexit std::exit
#endif

#endif /* KFMMessaging_H__ */
