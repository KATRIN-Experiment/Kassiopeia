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


using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const int Degree = 2;
    const int Size = (Degree + 1)*(Degree + 1);

    std::cout<<std::setprecision(4);

    double P[3] = {1.3945732, 2.045345, 4.0353};
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
    double O2[3] = {170.23432, -0.34, 0.634534};

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


    std::cout<<"multipole radius = "<<mRadius<<std::endl;
    std::cout<<"translation radius = "<<Radius<<std::endl;
    std::cout<<"relative radius = "<<Radius/mRadius<<std::endl;

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
            double inv_Ajk = 1.0/KFMMath::A_Coefficient(k,j);
//            ResponseFunctions[si] *= inv_Ajk;
//            ResponseFunctions[si] *= Ipow[std::abs(k)%4];
        }
    }

    //compute the multipole pre-scaling factors
    std::vector< std::complex<double> > MultipoleScaleFactors;
    MultipoleScaleFactors.resize(Size);
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            double Amn = KFMMath::A_Coefficient(m,n);
            MultipoleScaleFactors[n*(n+1) + m ] = 1.0;//negIpow[std::abs(m)%4]*(Amn);
        }
    }

    //compute the local coefficient post-scaling factors
    std::vector< std::complex<double> > LocalScaleFactors;
    LocalScaleFactors.resize(Size);
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            double Amn =1.0;// KFMMath::A_Coefficient(m,n);
            LocalScaleFactors[n*(n+1) + m ] = 1.0;//negIpow[std::abs(m)%4]*(Amn);
        }
    }

    //now we prescale the multipoles
    std::vector< std::complex<double> > ScaledMultipoleMoments;
    ScaledMultipoleMoments.resize(Size);
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            ScaledMultipoleMoments[n*(n+1) + m] =  MultipoleMoments[n*(n+1) + m]*MultipoleScaleFactors[n*(n+1) + m];
        }
    }


    clock_t start, end;
    start = clock();
    double time;

    unsigned int N_iter = 1;

    for(unsigned int i=0; i<N_iter; i++)
    {

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

                    LocalMoments[tsi] += ResponseFunctions[rsi]*ScaledMultipoleMoments[ssi];
                }
            }
        }
    }

    }


    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds

    std::cout<<"Time for O(p^4) calculation = "<<time/((double)N_iter)<<std::endl;

    //now we post scale the local coefficients
    //std::cout<<"------------------------------------------------"<<std::endl;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            //std::cout<<"Unscaled Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
            LocalMoments[n*(n+1) + m] *= LocalScaleFactors[n*(n+1) + m];
        }
    }

    //std::cout<<"------------------------------------------------"<<std::endl;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            //std::cout<<"Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
        }
    }


    //now we want to try doing the M2L conversion using FFT to execute the cross-correlations

    //first we need to allocate matrices that are large enough to store the input/results
    int NRows = 2*Degree + 1;//3*Degree + 2;
    int NCols = 2*Degree2 + 1;
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

    //initialize the multipole moment matrix
  //  double mitigate_roundoff = std::sqrt((double)Degree/4.0);
  //  double mitigate_roundoff = 1.0;
//    double mitigate_roundoff = std::sqrt( ((2.0*Degree)/std::exp(1.0))*std::pow(4.0*M_PI*Degree, 1.0/(4.0*Degree)) )*(Radius);


    //This should have some small prefactor which scales with Degree
    //such that it is about sqrt(2) for p=8, about 1.0 for p=16 and about 0.5 for p=32
//    double mitigate_roundoff = (1.0/std::sqrt(Radius))*((2.0*Degree)/std::exp(1.0))*std::pow(4.0*M_PI*Degree, 1.0/(2.0*Degree));//*(Radius);

  //  double ratio = std::abs(MultipoleMoments[Degree*(Degree+1)])/std::abs(MultipoleMoments[0]);
 //   double mitigate_roundoff = std::sqrt(1.0/std::pow(ratio, 1.0/((double)Degree)));
  double mitigate_roundoff = 1.0;//  (1.0/Radius);//*std::pow( KFMMath::SqrtFactorial(4.0*Degree), 1.0/(2.0*Degree+1.0) );
  //  double mitigate_roundoff =  std::pow( KFMMath::SqrtFactorial(4.0*Degree), 1.0/(2.0*Degree+1.0) );


    std::cout<<"mitigation factor = "<<mitigate_roundoff<<std::endl;


    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            double Amn = KFMMath::A_Coefficient(m,n);
           // std::cout<<"Re-Scaled Factors ("<<n<<", "<<m<<") = "<<negIpow[std::abs(m)%4]*(Amn)*std::pow(-1.0, n)*std::pow(mitigate_roundoff, n+1)<<std::endl;
        }
    }


//    double mitigate_roundoff = std::pow( (double)Degree, 0.33 );


    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            MultipoleMomentMatrix[n][n+m] = std::pow(mitigate_roundoff, n+1)*ScaledMultipoleMoments[n*(n+1) + m];
        }
    }


    //initialize the response function matrix
    for(int j=0; j<=Degree2; j++)
    {
        for(int k = -j; k <= j; k++)
        {
            int si = j*(j+1) + k;
//            ResponseFunctionMatrix[j][Degree2+k] = ResponseFunctions[si];
            ResponseFunctionMatrix[j][j+k] = ResponseFunctions[si]*std::pow(mitigate_roundoff, -j-1);
            //std::cout<<"Re-Scaled Response ("<<j<<", "<<k<<") = "<<ResponseFunctionMatrix[j][k+j]<<std::endl;
        }
    }


//    std::cout<<"------------------------------------------------"<<std::endl;
//    std::cout<<"multipole matrix:"<<std::endl;
//    //print the moment moment matrix
//    for(int i=0; i<NRows; i++)
//    {
//        for(int j=0; j<NCols; j++)
//        {
//            std::cout<<MultipoleMomentMatrix[i][j]<<", ";
//        }
//        std::cout<<std::endl;
//    }

    std::cout<<"------------------------------------------------"<<std::endl;
    std::cout<<"response matrix:"<<std::endl;
    //print the moment moment matrix
    for(int i=0; i<NRows; i++)
    {
        for(int j=0; j<NCols; j++)
        {
            std::cout<<ResponseFunctionMatrix[i][j]<<", ";
        }
        std::cout<<std::endl;
    }


    //now perform the cross-correlation over the rows of the multipole/response matrices
    KFMFastFourierTransform* row_fft = new KFMFastFourierTransform();
    row_fft->SetSize(NCols);
    row_fft->SetForward();

    KFMFastFourierTransform* row_ifft = new KFMFastFourierTransform();
    row_ifft->SetSize(NCols);
    row_ifft->SetBackward();

    //now perform the cross-correlation over the columns of the multipole/response matrices
    KFMFastFourierTransform* col_fft = new KFMFastFourierTransform();
    col_fft->SetSize(NRows);
    col_fft->SetForward();

    KFMFastFourierTransform* col_ifft = new KFMFastFourierTransform();
    col_ifft->SetSize(NRows);
    col_ifft->SetBackward();


    //temp space for output
    std::vector< std::complex<double> > row_moment; row_moment.resize(NCols);
    std::vector< std::complex<double> > row_moment_dft; row_moment_dft.resize(NCols);
    std::vector< std::complex<double> > row_response; row_response.resize(NCols);
    std::vector< std::complex<double> > row_response_dft; row_response_dft.resize(NCols);
    std::vector< std::complex<double> > row_result; row_result.resize(NCols);
    std::vector< std::complex<double> > row_result_dft; row_result_dft.resize(NCols);

    //array wrappers
    unsigned int row_dim[1]; row_dim[0] = NCols;
    KFMArrayWrapper< std::complex<double>, 1> row_moment_wrapper( &(row_moment[0]), row_dim);
    KFMArrayWrapper< std::complex<double>, 1> row_moment_dft_wrapper( &(row_moment_dft[0]), row_dim);
    KFMArrayWrapper< std::complex<double>, 1> row_response_wrapper( &(row_response[0]), row_dim);
    KFMArrayWrapper< std::complex<double>, 1> row_response_dft_wrapper( &(row_response_dft[0]), row_dim);
    KFMArrayWrapper< std::complex<double>, 1> row_result_wrapper( &(row_result[0]), row_dim);
    KFMArrayWrapper< std::complex<double>, 1> row_result_dft_wrapper( &(row_result_dft[0]), row_dim);

    //temp space for output
    std::vector< std::complex<double> > col_moment; col_moment.resize(NRows);
    std::vector< std::complex<double> > col_moment_dft; col_moment_dft.resize(NRows);
    std::vector< std::complex<double> > col_response; col_response.resize(NRows);
    std::vector< std::complex<double> > col_response_dft; col_response_dft.resize(NRows);
    std::vector< std::complex<double> > col_result; col_result.resize(NRows);
    std::vector< std::complex<double> > col_result_dft; col_result_dft.resize(NRows);

    //array wrappers
    unsigned int col_dim[1]; col_dim[0] = NRows;
    KFMArrayWrapper< std::complex<double>, 1> col_moment_wrapper( &(col_moment[0]), col_dim);
    KFMArrayWrapper< std::complex<double>, 1> col_moment_dft_wrapper( &(col_moment_dft[0]), col_dim);
    KFMArrayWrapper< std::complex<double>, 1> col_response_wrapper( &(col_response[0]), col_dim);
    KFMArrayWrapper< std::complex<double>, 1> col_response_dft_wrapper( &(col_response_dft[0]), col_dim);
    KFMArrayWrapper< std::complex<double>, 1> col_result_wrapper( &(col_result[0]), col_dim);
    KFMArrayWrapper< std::complex<double>, 1> col_result_dft_wrapper( &(col_result_dft[0]), col_dim);



////////////////////////////////////////////////////////////////////////////////

    for(int i=0; i<NRows; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NCols; j++)
        {
            row_response_wrapper[j] = ResponseFunctionMatrix[i][j];
        }

        //backwards fft the response coeff row
        row_ifft->SetInput(&(row_response_wrapper));
        row_ifft->SetOutput(&(row_response_dft_wrapper));

        row_ifft->Initialize();
        row_ifft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NCols; j++)
        {
            ResponseFunctionMatrix[i][j] = row_response_dft_wrapper[j];
        }
    }

////////////////////////////////////////////////////////////////////////////////




    //perform the column transforms
    for(int i=0; i<NCols; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NRows; j++)
        {
            col_response_wrapper[j] = ResponseFunctionMatrix[j][i];
        }


        //forward fft the response coeff column
        col_fft->SetInput(&(col_response_wrapper));
        col_fft->SetOutput(&(col_response_dft_wrapper));
        col_fft->Initialize();
        col_fft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NRows; j++)
        {
            ResponseFunctionMatrix[j][i] = col_response_dft_wrapper[j];
        }
    }



////////////////////////////////////////////////////////////////////////////////


    double norm = (1.0/(double)(NCols*NRows));



    start = clock();

    for(unsigned int i=0; i<N_iter; i++)
    {


    //std::cout<<"------------------------------------------------"<<std::endl;
    //compute the row transforms
    for(int i=0; i< (Degree+1) ; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NCols; j++)
        {
            row_moment_wrapper[j] = MultipoleMomentMatrix[i][j];
            //row_response_wrapper[j] = ResponseFunctionMatrix[i][j];
        }

        //forward fft the multipole moment row
        row_fft->SetInput(&row_moment_wrapper);
        row_fft->SetOutput(&row_moment_dft_wrapper);

        row_fft->Initialize();
        row_fft->ExecuteOperation();

//        //backwards fft the response coeff row
//        row_ifft->SetInput(&(row_response_wrapper));
//        row_ifft->SetOutput(&(row_response_dft_wrapper));

//        row_ifft->Initialize();
//        row_ifft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NCols; j++)
        {
            MultipoleMomentMatrix[i][j] = row_moment_dft_wrapper[j];
//            ResponseFunctionMatrix[i][j] = row_response_dft_wrapper[j];
        }
    }

//    std::cout<<"------------------------------------------------"<<std::endl;
//    std::cout<<"multipole matrix after row transforms:"<<std::endl;
//    //print the moment moment matrix
//    for(int i=0; i<NRows; i++)
//    {
//        for(int j=0; j<NCols; j++)
//        {
//            std::cout<<MultipoleMomentMatrix[i][j]<<", ";
//        }
//        std::cout<<std::endl;
//    }


////////////////////////////////////////////////////////////////////////////////







    //perform the column transforms
    for(int i=0; i<NCols; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NRows; j++)
        {
            col_moment_wrapper[j] = MultipoleMomentMatrix[j][i];
//            col_response_wrapper[j] = ResponseFunctionMatrix[j][i];
        }

        //backwards fft the multipole moment column
        col_ifft->SetInput(&col_moment_wrapper);
        col_ifft->SetOutput(&col_moment_dft_wrapper);
        col_ifft->Initialize();
        col_ifft->ExecuteOperation();

//        //forward fft the response coeff column
//        col_fft->SetInput(&(col_response_wrapper));
//        col_fft->SetOutput(&(col_response_dft_wrapper));
//        col_fft->Initialize();
//        col_fft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NRows; j++)
        {
            MultipoleMomentMatrix[j][i] = col_moment_dft_wrapper[j];
//            ResponseFunctionMatrix[j][i] = col_response_dft_wrapper[j];
        }
    }

    //now we perform the pointwise multiplication
    for(unsigned int i=0; i<NRows; i++)
    {
        for(unsigned int j=0; j<NCols; j++)
        {
            LocalMomentMatrix[i][j] = MultipoleMomentMatrix[i][j]*ResponseFunctionMatrix[i][j];
        }
    }


    //now we have to idft the local moment matrix

    //compute the row transforms
    for(int i=0; i<NRows; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NCols; j++)
        {
            row_moment_wrapper[j] = LocalMomentMatrix[i][j];
        }

        //backwards fft the response coeff row
        row_ifft->SetInput(&(row_moment_wrapper));
        row_ifft->SetOutput(&(row_moment_dft_wrapper));

        row_ifft->Initialize();
        row_ifft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NCols; j++)
        {
            LocalMomentMatrix[i][j] = row_moment_dft_wrapper[j];
        }
    }



    //perform the column transforms
    for(int i=0; i<NCols; i++)
    {
        //copy moment row and response row into temp space
        for(int j=0; j<NRows; j++)
        {
            col_moment_wrapper[j] = LocalMomentMatrix[j][i];
        }

        //backwards fft the multipole moment column
        col_ifft->SetInput(&col_moment_wrapper);
        col_ifft->SetOutput(&col_moment_dft_wrapper);
        col_ifft->Initialize();
        col_ifft->ExecuteOperation();

        //copy results back
        for(int j=0; j<NRows; j++)
        {
            LocalMomentMatrix[j][i] = norm*col_moment_dft_wrapper[j];
        }
    }

    }

    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds

    std::cout<<"Time for O(p^2*log(p)) calculation = "<<time/((double)N_iter)<<std::endl;

//    std::cout<<"------------------------------------------------"<<std::endl;
//    std::cout<<"local coeff matrix:"<<std::endl;
//    //print the moment moment matrix
//    for(int i=0; i<NRows; i++)
//    {
//        for(int j=0; j<NCols; j++)
//        {
//            std::cout<<LocalMomentMatrix[i][j]<<", ";
//        }
//        std::cout<<std::endl;
//    }

    //now we want to extract the unscaled local coefficient from the matrix
    for(int n=0; n<=Degree; n++)
    {
        int index = 0;
        for(int m=n; m>=-n; m--)
        {
            LocalMoments[n*(n+1)+m] = LocalMomentMatrix[n][(index + NCols)%NCols];
            index--;
        }
    }

    //now we post scale the local coefficients
//    std::cout<<"------------------------------------------------"<<std::endl;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
           // std::cout<<"Unscaled Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
            LocalMoments[n*(n+1) + m] *= std::pow(mitigate_roundoff, n)*LocalScaleFactors[n*(n+1) + m];
        }
    }



    //std::cout<<"------------------------------------------------"<<std::endl;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
           // std::cout<<"Local Moment("<<n<<", "<<m<<") = "<<LocalMoments[ n*(n+1) + m]<<std::endl;
        }
    }


    //std::cout<<"------------------------------------------------"<<std::endl;
    double l2_norm_err = 0.0;
    double l2_norm = 0.0;
    for(int n=0; n<=Degree; n++)
    {
        for(int m=-n; m<=n; m++)
        {
            std::complex<double> moment = 0.5*( LocalMoments[n*(n+1) + m] + std::conj(LocalMoments[n*(n+1) - m]) );
            std::complex<double> diff = moment - OriginalLocalMoments[n*(n+1) + m];

            //std::cout<<"Abs Error ("<<n<<", "<<m<<") = "<<std::sqrt(std::real(diff*std::conj(diff)))<<std::endl;
            l2_norm_err += std::real(diff*std::conj(diff));
            l2_norm += std::real( OriginalLocalMoments[n*(n+1) + m]*std::conj(OriginalLocalMoments[n*(n+1) + m]) );
        }
    }


    std::cout<<"rows*colums = "<<NRows*NCols*std::log(NRows*NCols)<<std::endl;
    std::cout<<"p4 = "<<Degree*Degree*Degree*Degree<<std::endl;

    std::cout<<"l2_norm error = "<<std::sqrt(l2_norm_err)<<std::endl;
    std::cout<<"norm = "<<std::sqrt(l2_norm)<<std::endl;
    std::cout<<"relative l2 norm error = "<<std::sqrt(l2_norm_err/l2_norm)<<std::endl;


    return 0;
}
