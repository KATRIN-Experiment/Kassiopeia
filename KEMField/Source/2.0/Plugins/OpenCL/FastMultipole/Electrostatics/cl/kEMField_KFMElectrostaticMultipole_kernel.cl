#ifndef KFMElectrostaticMultipole_Kernel_Defined_H
#define KFMElectrostaticMultipole_Kernel_Defined_H

#include "kEMField_defines.h"

#include "kEMField_KFMTriangleMultipole.cl"
#include "kEMField_KFMRectangleMultipole.cl"
#include "kEMField_KFMLineSegmentMultipole.cl"
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
}


__kernel void
ElectrostaticMultipole( const unsigned int n_elements, //number of primitives to process
                        __constant const CL_TYPE* a_coefficient,
                        __constant const CL_TYPE* equatorial_plm,
                        __constant const CL_TYPE* axial_plm,
                        __constant const CL_TYPE* pinchon_j,
                        __global const CL_TYPE4* origin_input,
                        __global const CL_TYPE16* vertex_input,
                        __global const CL_TYPE* basis_input,
                        __global CL_TYPE2* moment_output)
{

    // Get the index of the current element to be processed
    unsigned int i_global = get_global_id(0);

    if(i_global < n_elements)
    {
        //copy the global data into private workspace
        //retrieve the origin data
        CL_TYPE4 target_origin = origin_input[i_global];
        CL_TYPE4 source_origin;
        CL_TYPE16 vertex = vertex_input[i_global]; //retrieve the vertex data
        CL_TYPE total_charge = basis_input[i_global]; //retrieve the basis data

        //private workspace for computation of multipole moments
        CL_TYPE2 source_moments[KFM_REAL_STRIDE];
        CL_TYPE2 target_moments[KFM_REAL_STRIDE];
        CL_TYPE scratch1[KFM_COMPLEX_STRIDE];
        CL_TYPE scratch2[KFM_COMPLEX_STRIDE];

        //compute the multipole moments and fill the scratch space array
        int element_type = DetermineElementType(vertex.s3, vertex.s7);

        if(element_type == KFM_LINE_TYPE)
        {
            source_origin =
            LineSegmentMultipoleMoments(KFM_DEGREE, vertex, scratch1, source_moments);
        }

        if(element_type == KFM_TRIANGLE_TYPE)
        {
            source_origin =
            TriangleMultipoleMoments(KFM_DEGREE, equatorial_plm, pinchon_j, vertex, scratch1, scratch2, source_moments);
        }

        if(element_type == KFM_RECTANGLE_TYPE)
        {
            //here we use target moments as scratch space
            source_origin =
            RectangleMultipoleMoments(KFM_DEGREE, equatorial_plm, pinchon_j, vertex, scratch1, scratch2, target_moments, source_moments);
        }

        //translate moments to be an expansion about the target origin
        TranslateMultipoleMomentsFast(KFM_DEGREE,
                                      a_coefficient,
                                      axial_plm,
                                      pinchon_j,
                                      source_origin,
                                      target_origin,
                                      scratch1,
                                      scratch2,
                                      source_moments,
                                      target_moments);


        //copy moments into the global output and scale by charge
        for(int i=0; i<KFM_REAL_STRIDE; i++)
        {
            moment_output[i_global*KFM_REAL_STRIDE + i] = total_charge*target_moments[i];
//            moment_output[i_global*KFM_REAL_STRIDE + i] = target_moments[i];
        }

    }

}

#endif
