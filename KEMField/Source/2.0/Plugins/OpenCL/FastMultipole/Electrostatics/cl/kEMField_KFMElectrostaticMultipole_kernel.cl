#ifndef KFMElectrostaticMultipole_Kernel_Defined_H
#define KFMElectrostaticMultipole_Kernel_Defined_H

#include "kEMField_defines.h"

#include "kEMField_KFMTriangleMultipole.cl"
#include "kEMField_KFMTriangleMultipoleNumerical.cl"
#include "kEMField_KFMRectangleMultipole.cl"
#include "kEMField_KFMRectangleMultipoleNumerical.cl"
#include "kEMField_KFMLineSegmentMultipole.cl"
#include "kEMField_KFMLineSegmentMultipoleNumerical.cl"
#include "kEMField_KFMMultipoleTranslation.cl"

//these constants are defined at compile time
//KFM_DEGREE
//KFM_COMPLEX_STRIDE
//KFM_REAL_STRIDE

//NOTE: need to delete the folder ~/.nv/ComputeCache to force recompile on nvidia machines!!

#define KFM_POINT_TYPE 0
#define KFM_LINE_TYPE 1
#define KFM_TRIANGLE_TYPE 2
#define KFM_RECTANGLE_TYPE 3

int
DetermineElementType(CL_TYPE msb, CL_TYPE lsb)
{
    if( (msb < 0) && (lsb < 0) ) //point type
    {
        return KFM_POINT_TYPE;
    }

    if( (msb < 0) && (lsb > 0) ) //line type
    {
        return KFM_LINE_TYPE;
    }

    if( (msb > 0) && (lsb < 0) ) //triangle type
    {
        return KFM_TRIANGLE_TYPE;;
    }

    if( (msb > 0) && (lsb > 0) ) //rectangle type
    {
        return KFM_RECTANGLE_TYPE;
    }

    return KFM_POINT_TYPE;
}

__kernel void
ElectrostaticMultipole( const unsigned int n_elements, //number of primitives to process
                        __constant const CL_TYPE* a_coefficient,
                        __constant const CL_TYPE* equatorial_plm,
                        __constant const CL_TYPE* axial_plm,
                        __constant const CL_TYPE* pinchon_j,
                        __constant const CL_TYPE* abscissa,
                        __constant const CL_TYPE* weights,
                        __global const CL_TYPE4* origin_input,
                        __global const CL_TYPE16* vertex_input,
                        __global const CL_TYPE* basis_input,
                        __global CL_TYPE2* moment_output)
{

    // Get the index of the current element to be processed
    unsigned int i_global = get_global_id(0);

    //private workspace for computation of multipole moments
    CL_TYPE2 source_moments[KFM_REAL_STRIDE];
    CL_TYPE2 target_moments[KFM_REAL_STRIDE];

    //allocate scratch space in one block
    CL_TYPE scratch1[KFM_COMPLEX_STRIDE];
    CL_TYPE scratch2[KFM_COMPLEX_STRIDE];

    CL_TYPE4 target_origin;
    CL_TYPE4 source_origin;
    CL_TYPE16 vertex;
    int element_type, offset;

    //copy the global data into private workspace
    //retrieve the origin data
    target_origin = origin_input[i_global];
    vertex = vertex_input[i_global]; //retrieve the vertex data
    CL_TYPE total_charge = basis_input[i_global]; //retrieve the basis data
    //compute the multipole moments and fill the scratch space array
    element_type = DetermineElementType(vertex.s3, vertex.s7);
    offset = i_global*KFM_REAL_STRIDE;

    if(i_global < n_elements)
    {
        if(element_type == KFM_LINE_TYPE)
        {
            source_origin =
            LineSegmentMultipoleMoments(KFM_DEGREE, vertex, scratch1, source_moments);
            TranslateMultipoleMomentsFast(KFM_DEGREE, a_coefficient, axial_plm, pinchon_j, source_origin,
                                              target_origin, scratch1, scratch2, source_moments, target_moments);
            //LineSegmentMultipoleMomentsNumerical(KFM_DEGREE, target_origin, vertex, abscissa, weights, target_moments);
        }

        if(element_type == KFM_TRIANGLE_TYPE)
        {
            if(vertex.sB < 0)
            {
                source_origin =
                TriangleMultipoleMoments(KFM_DEGREE, equatorial_plm, pinchon_j, vertex, scratch1, scratch2, source_moments);
                //translate moments to be an expansion about the target origin
                TranslateMultipoleMomentsFast(KFM_DEGREE, a_coefficient, axial_plm, pinchon_j, source_origin,
                                              target_origin, scratch1, scratch2, source_moments, target_moments);
            }
            else
            {
                //aspect ratio (too large) disqualifies use of analytic method
                TriangleMultipoleMomentsNumerical(KFM_DEGREE, target_origin, vertex, abscissa, weights, target_moments);
            }
        }

        if(element_type == KFM_RECTANGLE_TYPE)
        {
            if(vertex.sB < 0)
            {
                //here we use target moments as scratch space
                source_origin =
                RectangleMultipoleMoments(KFM_DEGREE, equatorial_plm, pinchon_j, vertex, scratch1, scratch2, target_moments, source_moments);
                //translate moments to be an expansion about the target origin
                TranslateMultipoleMomentsFast(KFM_DEGREE, a_coefficient, axial_plm, pinchon_j, source_origin,
                                              target_origin, scratch1, scratch2, source_moments, target_moments);
            }
            else
            {
                //aspect ratio (too large) disqualifies use of analytic method
                RectangleMultipoleMomentsNumerical(KFM_DEGREE, target_origin, vertex, abscissa, weights, target_moments);
            }
        }


    }

    //these local barriers are only necessary for running on intel MIC cards, since they force a scalar version
    //of this kernel to run, the auto-vectorized version of this kernel results in an illegal memory access
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i_global < n_elements)
    {
        //copy moments into the global output and scale by charge
        for(int i=0; i<KFM_REAL_STRIDE; i++)
        {
            moment_output[offset + i] = total_charge*target_moments[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

}


#endif
