#include <iostream>
#include <cmath>
#include <iomanip>

#include <complex>

#include "KFMArrayMath.hh"

#include "KFMFastFourierTransform.hh"

#include "KFMMomentTransformer.hh"
#include "KFMMomentTransformerTypes.hh"

#include "KFMMath.hh"
#include "KFMArrayWrapper.hh"
#include "KFMMatrixOperations.hh"
#include "KFMVectorOperations.hh"


using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const int Degree = 1;
    const int Size = (Degree + 1)*(Degree + 1);

    std::cout<<std::setprecision(4);

    double P[3] = {0.0945732, -0.345345, 0.1353};
    double O[3] = {0.0, 0.0, 0.0};
    double Delta[3];
    for(unsigned int i=0; i<3; i++){Delta[i] = P[i] - O[i];};


    double mRadius = KFMMath::Radius(O,P);

    std::vector< std::complex<double> > MultipoleMoments;
    std::vector< std::complex<double> > LocalMoments;

    MultipoleMoments.resize(Size);
    LocalMoments.resize(Size);

    //we want to create some multipole moments for the point P
    KFMMath::RegularSolidHarmonic_Cart_Array(Degree, Delta, &(MultipoleMoments[0]) );

    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
           // std::cout<<"Multipole Moment("<<n<<", "<<m<<") = "<<MultipoleMoments[ n*(n+1) + m]<<std::endl;
        }
    }

    //define the origin about which we want to create the new local coefficient expansion
//    double O2[3] = {-2.325, 3.345346, 7.43451};
    double alpha = 0.2345;
    double x_val = std::cos(alpha);
    double y_val = std::sin(alpha);
    double O2[3] = {2.454, -1.23423, 3.4234};

    KFMMomentTransformer_3DLaplaceM2L M2LTransformer;
    M2LTransformer.SetNumberOfTermsInSeries(Size);
    M2LTransformer.Initialize();

    M2LTransformer.SetSourceOrigin(O);
    M2LTransformer.SetTargetOrigin(O2);
    M2LTransformer.SetSourceMoments(&MultipoleMoments);
    M2LTransformer.Transform();
    M2LTransformer.GetTargetMoments(&LocalMoments);

    std::vector< std::complex<double> > OriginalLocalMoments;
    OriginalLocalMoments.resize(Size);

    //std::cout<<"------------------------------------------------"<<std::endl;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            std::cout<<"Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
            OriginalLocalMoments[ n*(n+1) + m] = LocalMoments[ n*(n+1) + m];
        }
    }


    //now we want to test computing the local coefficients from the multipole coefficients through cross-correlation

    //first we want to compute the response functions, for this we need all of the irregular solid harmonics
    //from n=0 up to n=2*Degree
    int Degree2 = 2*Degree;
    int Size2 = (Degree2+1)*(Degree2+1);
    std::vector< std::complex<double> > ResponseFunctions;
    ResponseFunctions.resize(Size2);

    //compute part of the response functions
    for(unsigned int i=0; i<3; i++){Delta[i] = O[i] - O2[i];};
    double Radius = KFMMath::Radius(O,O2);
    KFMMath::IrregularSolidHarmonic_Cart_Array(Degree2, Delta, &(ResponseFunctions[0]) );


    //now compute the normalization factors
    std::vector< double > NormalizationFactors;
    NormalizationFactors.resize(Size2);

    for(int j=0; j<=Degree2; j++)
    {
        for(int k = -j; k <= j; k++)
        {
            NormalizationFactors[ j*(j+1) + k ] = KFMMath::A_Coefficient(k,j);
        }
    }


    //apply normalization factor to response functions
    std::complex<double> I(0.0, 1.0);
    std::complex<double> negIpow[4];
    std::complex<double> Ipow[4];
    negIpow[0] = std::complex<double>(1.0,0.0);
    negIpow[1] = std::complex<double>(0.0,-1.0);
    negIpow[2] = std::complex<double>(-1.0, 0.0);
    negIpow[3] = std::complex<double>(0.0,1.0);

    Ipow[0] = std::complex<double>(1.0,0.0);
    Ipow[1] = std::complex<double>(0.0,1.0);
    Ipow[2] = std::complex<double>(-1.0, 0.0);
    Ipow[3] = std::complex<double>(0.0,-1.0);

    for(int j=0; j<=Degree2; j++)
    {
        for(int k = -j; k <= j; k++)
        {
            int si = j*(j+1) + k;
            int index = std::abs(k) % 4;
            ResponseFunctions[si] *=  Ipow[index]/NormalizationFactors[si];
        }
    }

    //compute the multipole pre-scaling factors
    std::vector< std::complex<double> > MultipoleScaleFactors;
    MultipoleScaleFactors.resize(Size);
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            int si = n*(n+1) + m;
            MultipoleMoments[si] *= std::pow(-1.0, n)*NormalizationFactors[si]*std::pow(-1.0*I, std::fabs(m));
        }
    }


    //now we want to try doing the M2L conversion using FFT to execute the cross-correlations

    //first we need to allocate matrices that are large enough to store the input/results
    int NRows = (Degree + 1)*(Degree+1);
    int NCols = (Degree + 1)*(Degree+1);
    std::vector< std::vector< std::complex<double> > > MultipoleMomentMatrix;
    std::vector< std::vector< std::complex<double> > > ResponseFunctionMatrix;
    std::vector< std::vector< std::complex<double> > > LocalMomentMatrix;
    MultipoleMomentMatrix.resize(NRows);
    ResponseFunctionMatrix.resize(NRows);
    LocalMomentMatrix.resize(NRows);
    for(int i=0; i<NRows; i++)
    {
        MultipoleMomentMatrix[i].resize(NCols);
        ResponseFunctionMatrix[i].resize(NCols);
        LocalMomentMatrix[i].resize(NCols);
    }

    for(int i=0; i<NRows; i++)
    {
        for(int j=0; j<NCols; j++)
        {
            MultipoleMomentMatrix[i][j] = std::complex<double>(0,0);
            ResponseFunctionMatrix[i][j] = std::complex<double>(0,0);
            LocalMomentMatrix[i][j] = std::complex<double>(0,0);
        }
    }

    clock_t start, end;
    start = clock();
    double time;


    //first we do this the slow O(p^4) way
    int tsi, ssi, rsi;
    int dsum, odiff;
    for(int j=0; j<=Degree; j++)
    {
        for(int k=-j; k<=j; k++)
        {
            tsi = j*(j+1) + k;
            LocalMoments[tsi] = std::complex<double>(0.0,0.0);

            for(int n=0; n<=Degree; n++)
            {
                for(int m=-n; m<=n; m++)
                {
                    dsum = j+n;
                    odiff = m-k;

                    ssi = n*(n+1) + m;
                    rsi = dsum*(dsum+1) + odiff;

                    if(j+n <= 2*Degree)
                    {

//                        double norm = ( (NormalizationFactors[ssi]*NormalizationFactors[tsi])/NormalizationFactors[rsi] );
//                        double pm = norm;

//                        if(n%2 == 1){pm *= -1.0;};

//                        if( k*m > 0)
//                        {
//                            if( ((unsigned int)std::min(std::fabs(m), std::fabs(k)))%2 == 1){pm*= -1.0;}
//                        }

//                        //std::cout<<"Norm("<<j<<", "<<k<<","<<n<<","<<m<<") = "<<norm<<std::endl;
//                        //std::cout<<norm<<std::endl;

//                        LocalMoments[tsi] += pm*ResponseFunctions[rsi]*MultipoleMoments[ssi];


                        LocalMoments[tsi] += ResponseFunctions[rsi]*MultipoleMoments[ssi];

                        ResponseFunctionMatrix[tsi][ssi] = ResponseFunctions[rsi];
                    }




                }
            }
        }
    }



    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds

    std::cout<<"Time for O(p^4) calculation = "<<time<<std::endl;



    //compute the local coefficient post-scaling factors
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            int si = n*(n+1) + m;
            //LocalMoments[n*(n+1) + m ] *= std::pow(-1.0, n)*std::pow(-1.0*I, std::fabs(m));
            LocalMoments[si] *= NormalizationFactors[si]*std::pow(-1.0*I, std::fabs(m));
            std::cout<<"Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
        }
    }




    //std::cout<<"------------------------------------------------"<<std::endl;
    double l2_norm_err = 0.0;
    double l2_norm = 0.0;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            std::complex<double> diff = LocalMoments[n*(n+1) + m] - OriginalLocalMoments[n*(n+1) + m];

            //std::cout<<"Abs Error ("<<n<<", "<<m<<") = "<<std::sqrt(std::real(diff*std::conj(diff)))<<std::endl;
            l2_norm_err += std::real(diff*std::conj(diff));
            l2_norm += std::real( OriginalLocalMoments[n*(n+1) + m]*std::conj(OriginalLocalMoments[n*(n+1) + m]) );
        }
    }

    std::cout<<"l2_norm error = "<<std::sqrt(l2_norm_err)<<std::endl;
    std::cout<<"norm = "<<std::sqrt(l2_norm)<<std::endl;
    std::cout<<"relative l2 norm error = "<<std::sqrt(l2_norm_err/l2_norm)<<std::endl;

//    std::cout<<"------------------------------------------------"<<std::endl;
//    std::cout<<"response matrix:"<<std::endl;
//    //print the moment moment matrix
//    for(int i=0; i<NRows; i++)
//    {
//        for(int j=0; j<NCols; j++)
//        {
//            std::cout<<ResponseFunctionMatrix[i][j]<<", ";
//        }
//        std::cout<<std::endl;
//    }


    //now we want to figure out want the SVD of the reponse function matrix looks like
    kfm_matrix* R = kfm_matrix_calloc(2*NRows, 2*NCols);

    //now we are going to set the rows and columns according to the real and imag parts of R
    for(int i=0; i<NRows; i++)
    {
        for(int j=0; j<NCols; j++)
        {
            kfm_matrix_set(R, i, j, std::real(ResponseFunctionMatrix[i][j]) );
            kfm_matrix_set(R, i+NRows, j+NCols, std::real(ResponseFunctionMatrix[i][j]) );
            kfm_matrix_set(R, i+NRows, j, -1.0*std::imag(ResponseFunctionMatrix[i][j]) );
            kfm_matrix_set(R, i, j+NCols, std::imag(ResponseFunctionMatrix[i][j]) );
        }
    }

    kfm_matrix* U = kfm_matrix_calloc(2*NRows, 2*NCols);
    kfm_matrix* V = kfm_matrix_calloc(2*NRows, 2*NRows);
    kfm_matrix* S = kfm_matrix_calloc(2*NRows, 2*NRows);
    kfm_vector* s = kfm_vector_calloc(2*NRows);

    kfm_matrix_svd(R, U, s, V);

    for(int i=0; i<2*NRows; i++)
    {
        std::cout<<"singular value @ "<<i<<" = "<<kfm_vector_get(s, i)<<std::endl;
    }

    for(int i=0; i<2*NRows; i++)
    {
        for(int j=0; j<2*NCols; j++)
        {
            if(std::fabs(kfm_matrix_get(U,i,j)) < 1e-14){kfm_matrix_set(U,i,j, 0.0);};
        }
    }


    //now we print unitary matrix U

//    kfm_matrix_print(U);



    return 0;
}
