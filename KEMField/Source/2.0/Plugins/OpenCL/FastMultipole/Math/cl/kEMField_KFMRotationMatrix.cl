#ifndef KFMRotationMatrix_Defined_H
#define KFMRotationMatrix_Defined_H

#define KFM_EPSILON 1e-14

//rotation matrix storage index from (row,col)
int rmsi(int row, int col)
{
    return 3*row + col;
}



CL_TYPE4
EulerAnglesFromMatrix(CL_TYPE* R) //R is ptr to array of size 9
{
    CL_TYPE4 ret_val;
    CL_TYPE alpha, beta, gamma;

    //compute the product of the matrix with its transpose
    //then compute the L2 norm of the difference of this product
    //from the identity

    CL_TYPE tol = 0.0;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            CL_TYPE mx_element = 0.0;
            for(size_t k=0; k<3; k++)
            {
                mx_element += R[rmsi(i,k)]*R[rmsi(j,k)];
            }

            if(i != j)
            {
                tol += mx_element*mx_element;
            }
            else
            {
                tol += (mx_element - 1.0)*(mx_element - 1.0);
            }
        }
    }
    tol = sqrt(tol + KFM_EPSILON*KFM_EPSILON);


    //the angles that are computed are not unique but represent one possible
    //set of angles that construct the rotation matrix
    bool isDegenerate = false;

    //if  |1-|cos(beta)| | < tol we are in the degenerate case
    if( fabs( 1.0 - fabs( R[rmsi(2,2)] ) ) <= tol)
    {
        isDegenerate = true;
    }

    if(!isDegenerate)
    {
        beta = acos( R[rmsi(2,2)] );
        alpha = atan2( (-1.0*R[rmsi(2,1)] )/sin(beta), (R[rmsi(2,0)])/sin(beta) );
        gamma = atan2( (-1.0*R[rmsi(1,2)] )/sin(beta), -1.0*(R[rmsi(0,2)])/sin(beta)  ) ;
    }
    else
    {
        if( fabs(1.0 - R[rmsi(2,2)])  <= tol)
        {
            alpha =  atan2(R[rmsi(1,0)], R[rmsi(0,0)]);
            beta = 0.0;
            gamma = 0.0;
        }
        else if( fabs(1.0 + R[rmsi(2,2)]) <= tol)
        {
            alpha = atan2(R[rmsi(0,1)], R[rmsi(1,1)]);
            beta = M_PI;
            gamma = 0.0;
        }
        else
        {
            //either no solution found, or R is the identity!
            alpha = 0.0;
            beta = 0.0;
            gamma = 0.0;
        }
    }

    ret_val.s0 = alpha;
    ret_val.s1 = beta;
    ret_val.s2 = gamma;
    ret_val.s3 = 0.0;

    return ret_val;
}

//______________________________________________________________________________

CL_TYPE4
EulerAnglesFromMatrixTranspose(CL_TYPE* R) //R is ptr to array of size 9
{
    CL_TYPE4 ret_val;
    CL_TYPE alpha, beta, gamma;

    //compute the product of the matrix with its transpose
    //then compute the L2 norm of the difference of this product
    //from the identity

    CL_TYPE tol = 0.0;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            CL_TYPE mx_element = 0.0;
            for(size_t k=0; k<3; k++)
            {
                mx_element += R[rmsi(i,k)]*R[rmsi(j,k)];
            }

            if(i != j)
            {
                tol += mx_element*mx_element;
            }
            else
            {
                tol += (mx_element - 1.0)*(mx_element - 1.0);
            }
        }
    }
    tol = sqrt(tol + KFM_EPSILON*KFM_EPSILON);


    //the angles that are computed are not unique but represent one possible
    //set of angles that construct the rotation matrix
    bool isDegenerate = false;

    //if  |1-|cos(beta)| | < tol we are in the degenerate case
    if( fabs( 1.0 - fabs( R[rmsi(2,2)] ) ) <= tol)
    {
        isDegenerate = true;
    }

    if(!isDegenerate)
    {
        beta = acos( R[rmsi(2,2)] );
        alpha = atan2( (-1.0*R[rmsi(1,2)] )/sin(beta), (R[rmsi(0,2)])/sin(beta) );
        gamma = atan2( (-1.0*R[rmsi(2,1)] )/sin(beta), -1.0*(R[rmsi(2,0)])/sin(beta)  ) ;
    }
    else
    {
        if( fabs(1.0 - R[rmsi(2,2)])  <= tol)
        {
            alpha =  atan2(R[rmsi(0,1)], R[rmsi(0,0)]);
            beta = 0.0;
            gamma = 0.0;
        }
        else if( fabs(1.0 + R[rmsi(2,2)]) <= tol)
        {
            alpha = atan2(R[rmsi(1,0)], R[rmsi(1,1)]);
            beta = M_PI;
            gamma = 0.0;
        }
        else
        {
            //either no solution found, or R is the identity!
            alpha = 0.0;
            beta = 0.0;
            gamma = 0.0;
        }
    }

    ret_val.s0 = alpha;
    ret_val.s1 = beta;
    ret_val.s2 = gamma;
    ret_val.s3 = 0.0;

    return ret_val;
}


//______________________________________________________________________________


CL_TYPE4
EulerAnglesFromAxes(CL_TYPE4 x_axis, CL_TYPE4 y_axis, CL_TYPE4 z_axis)
{
//    CL_TYPE R[3][3];

//    R[0][0] = x_axis.s0;
//    R[0][1] = y_axis.s0;
//    R[0][2] = z_axis.s0;

//    R[1][0] = x_axis.s1;
//    R[1][1] = y_axis.s1;
//    R[1][2] = z_axis.s1;

//    R[2][0] = x_axis.s2;
//    R[2][1] = y_axis.s2;
//    R[2][2] = z_axis.s2;

    CL_TYPE R[9];

    R[rmsi(0,0)] = x_axis.s0;
    R[rmsi(0,1)] = y_axis.s0;
    R[rmsi(0,2)] = z_axis.s0;

    R[rmsi(1,0)] = x_axis.s1;
    R[rmsi(1,1)] = y_axis.s1;
    R[rmsi(1,2)] = z_axis.s1;

    R[rmsi(2,0)] = x_axis.s2;
    R[rmsi(2,1)] = y_axis.s2;
    R[rmsi(2,2)] = z_axis.s2;

    return EulerAnglesFromMatrix(R);
}

CL_TYPE16
MatrixFromAxisAngle(CL_TYPE cos_angle, CL_TYPE sin_angle, CL_TYPE4 axis) //R is ptr to 3x3 array
{
    CL_TYPE R[9];

    //form the outer product of the axis vector and put in into R
    R[rmsi(0,0)] = axis.s0*axis.s0;
    R[rmsi(0,1)] = axis.s0*axis.s1;
    R[rmsi(0,2)] = axis.s0*axis.s2;
    R[rmsi(1,0)] = axis.s1*axis.s0;
    R[rmsi(1,1)] = axis.s1*axis.s1;
    R[rmsi(1,2)] = axis.s1*axis.s2;
    R[rmsi(2,0)] = axis.s2*axis.s0;
    R[rmsi(2,1)] = axis.s2*axis.s1;
    R[rmsi(2,2)] = axis.s2*axis.s2;

    CL_TYPE t = 1.0 - cos_angle;

    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            R[rmsi(i,j)] *= t;
        }
    }

    for(size_t i=0; i<3; i++)
    {
        R[rmsi(i,i)] += cos_angle;
    }

    R[rmsi(0,1)] -= sin_angle*(axis.s2);
    R[rmsi(0,2)] += sin_angle*(axis.s1);
    R[rmsi(1,2)] -= sin_angle*(axis.s0);

    R[rmsi(1,0)] += sin_angle*(axis.s2);
    R[rmsi(2,0)] -= sin_angle*(axis.s1);
    R[rmsi(2,1)] += sin_angle*(axis.s0);

    CL_TYPE16 mx;
    mx.s0 = R[rmsi(0,0)];
    mx.s1 = R[rmsi(0,1)];
    mx.s2 = R[rmsi(0,2)];

    mx.s3 = R[rmsi(1,0)];
    mx.s4 = R[rmsi(1,1)];
    mx.s5 = R[rmsi(1,2)];

    mx.s6 = R[rmsi(2,0)];
    mx.s7 = R[rmsi(2,1)];
    mx.s8 = R[rmsi(2,2)];

    return mx;

}



#endif /* KFMRotationMatrix_Defined_H */
