#ifndef KFMKernelResponseArrayTypes_HH__
#define KFMKernelResponseArrayTypes_HH__

#include "KFMKernelResponseArray.hh"
#include "KFMReducedKernelResponseArray.hh"

#include "KFMResponseKernel_3DLaplaceL2L.hh"
#include "KFMResponseKernel_3DLaplaceM2L.hh"
#include "KFMResponseKernel_3DLaplaceM2M.hh"

namespace KEMField
{

/*
*
*@file KFMKernelResponseArrayTypes.hh
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Dec  4 10:41:48 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KFMKernelResponseArray< KFMResponseKernel_3DLaplaceM2M, false, 3 > KFMKernelResponseArray_3DLaplaceM2M; //inward response
typedef KFMKernelResponseArray< KFMResponseKernel_3DLaplaceM2L, true, 3 > KFMKernelResponseArray_3DLaplaceM2L; //outward response
typedef KFMKernelResponseArray< KFMResponseKernel_3DLaplaceL2L, true, 3> KFMKernelResponseArray_3DLaplaceL2L; //outward response

typedef KFMReducedKernelResponseArray< KFMResponseKernel_3DLaplaceM2M, false, 3 > KFMKernelReducedResponseArray_3DLaplaceM2M; //inward response
typedef KFMReducedKernelResponseArray< KFMResponseKernel_3DLaplaceM2L, true, 3 > KFMKernelReducedResponseArray_3DLaplaceM2L; //outward response
typedef KFMReducedKernelResponseArray< KFMResponseKernel_3DLaplaceL2L, true, 3 > KFMKernelReducedResponseArray_3DLaplaceL2L; //outward response

}

#endif /* KFMKernelResponseArrayTypes_H__ */
