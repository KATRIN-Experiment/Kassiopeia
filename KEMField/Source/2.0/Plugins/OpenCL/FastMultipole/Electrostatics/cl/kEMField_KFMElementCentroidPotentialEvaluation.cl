#ifndef KFMElementCentroidFieldEvaluation_Defined_H
#define KFMElementCentroidFieldEvaluation_Defined_H

#include  "kEMField_defines.h"

#include "kEMField_KFMRaggedElementLookUp.cl"
#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_KFMSphericalMultipoleExpansionEvaluation.cl"
#include "kEMField_KFMArrayMath.cl"

//compile time constants
//DEGREE
//MOMENT_STRIDE

__kernel void
ElementCentroidPotentialEvaluation(unsigned int n_elements,
                                   unsigned int divisions,
                                   CL_TYPE low_corner_x,
                                   CL_TYPE low_corner_y,
                                   CL_TYPE low_corner_z,
                                   CL_TYPE node_side_length,
                                   __global unsigned int* node_list_start_index, //read only
                                   __global unsigned int* node_list_size, //read only
                                   __global CL_TYPE* element_centroids, //read only
                                   __global CL_TYPE2* local_moments,
                                   __global CL_TYPE* centroid_potentials //write only
                                  )
{
    //look up thread id, corresponding to the local index of then element we need to process
    unsigned int i_global = get_global_id(0);

    if(i_global < n_elements)
    {

        //assign private variable the array dimensions
        unsigned int dim[3];
        unsigned int node_spatial_index[3];
        unsigned int div_scratch[3];

        unsigned int spatial_stride = 1;
        for(unsigned int i=0; i<3; i++)
        {
            dim[i] = divisions;
            spatial_stride *= divisions;
        }

        //look up the indices of the appropriate node and element
        uint2 elem_node_index = RaggedElementLookup(spatial_stride, i_global, node_list_start_index, node_list_size);

        //get the element's centroid
        CL_TYPE4 P;
        P.s0 = element_centroids[3*elem_node_index.s0];
        P.s1 = element_centroids[3*elem_node_index.s0 + 1];
        P.s2 = element_centroids[3*elem_node_index.s0 + 2];

        //now we compute the node's center (the expansion origin)
        CL_TYPE4 O;
        O.s0 = low_corner_x + node_side_length/2.0;
        O.s1 = low_corner_y + node_side_length/2.0;
        O.s2 = low_corner_z + node_side_length/2.0;

        RowMajorIndexFromOffset(3, elem_node_index.s1, dim, node_spatial_index, div_scratch);

        O.s0 += node_spatial_index[0]*node_side_length;
        O.s1 += node_spatial_index[1]*node_side_length;
        O.s2 += node_spatial_index[2]*node_side_length;

        //now we compute the element centroids's spherical coordinates w.r.t to the node center
        CL_TYPE4 sph_coord;
        sph_coord.s0 = Radius(O,P);
        sph_coord.s1 = CosTheta(O,P);
        sph_coord.s3 = sqrt( (1.0 - cos_theta)*(1.0 + cos_theta) ); //sin(theta)
        sph_coord.s4 = Phi(O,P);

        //now we need to retrieve the local coefficients of this node
        CL_TYPE2 moments[MOMENT_STRIDE];
        for(unsigned int si=0; si<MOMENT_STRIDE; si++)
        {
            moments[si] = local_moments[si*spatial_stride + elem_node_index.s1];
        }

        //finally we can evaluate the potential from the local coefficients
        centroid_potentials[elem_node_index.s0] += ElectricPotential(DEGREE, sph_coord, moments);

    }

}


#endif
