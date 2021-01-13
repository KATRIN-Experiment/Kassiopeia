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

using KFMKernelResponseArray_3DLaplaceM2M =
    KFMKernelResponseArray<KFMResponseKernel_3DLaplaceM2M, false, 3>;  //inward response
using KFMKernelResponseArray_3DLaplaceM2L =
    KFMKernelResponseArray<KFMResponseKernel_3DLaplaceM2L, true, 3>;  //outward response
using KFMKernelResponseArray_3DLaplaceL2L =
    KFMKernelResponseArray<KFMResponseKernel_3DLaplaceL2L, true, 3>;  //outward response

using KFMKernelReducedResponseArray_3DLaplaceM2M =
    KFMReducedKernelResponseArray<KFMResponseKernel_3DLaplaceM2M, false, 3>;  //inward response
using KFMKernelReducedResponseArray_3DLaplaceM2L =
    KFMReducedKernelResponseArray<KFMResponseKernel_3DLaplaceM2L, true, 3>;  //outward response
using KFMKernelReducedResponseArray_3DLaplaceL2L =
    KFMReducedKernelResponseArray<KFMResponseKernel_3DLaplaceL2L, true, 3>;  //outward response

}  // namespace KEMField

#endif /* KFMKernelResponseArrayTypes_H__ */
