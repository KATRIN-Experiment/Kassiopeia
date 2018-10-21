#include "KFMElectrostaticLocalCoefficientCalculatorNumeric.hh"

namespace KEMField
{


KFMElectrostaticLocalCoefficientCalculatorNumeric::KFMElectrostaticLocalCoefficientCalculatorNumeric()
{
    fNumInt1D = new KVMPathIntegral<2>();
    fNumInt2D = new KVMSurfaceIntegral<2>();

    fSolidHarmonicWrapper = new KVMFieldWrapper< KFMElectrostaticLocalCoefficientCalculatorNumeric,
                                                &KFMElectrostaticLocalCoefficientCalculatorNumeric::IrregularSolidHarmonic>(this, 3, 2);

    fLine = new KVMLineSegment();
    fTriangle = new KVMTriangularSurface();
    fRectangle = new KVMRectangularSurface();

    fNumInt1D->SetCurve(fLine);
    fNumInt1D->SetField(fSolidHarmonicWrapper);

    fNumInt2D->SetField(fSolidHarmonicWrapper);

    fSolidHarmonics = NULL;
}

KFMElectrostaticLocalCoefficientCalculatorNumeric::~KFMElectrostaticLocalCoefficientCalculatorNumeric()
{
    delete fNumInt1D;
    delete fNumInt2D;
    delete fLine;
    delete fTriangle;
    delete fRectangle;
    delete[] fSolidHarmonics;
}


void KFMElectrostaticLocalCoefficientCalculatorNumeric::SetDegree(int l_max)
{
    fDegree = std::abs(l_max);
    fSize = (fDegree+1)*(fDegree + 1);
    fMoments.resize(fSize);
    if(fSolidHarmonics){delete[] fSolidHarmonics;};
    fSolidHarmonics = new std::complex<double>[fSize];
}

void KFMElectrostaticLocalCoefficientCalculatorNumeric::SetNumberOfQuadratureTerms(unsigned int n)
{
    fNumInt1D->SetNTerms(n);
    fNumInt2D->SetNTerms(n);
}

bool
KFMElectrostaticLocalCoefficientCalculatorNumeric::ConstructExpansion(double* target_origin,
                                                               const KFMPointCloud<3>* vertices,
                                                                KFMScalarMultipoleExpansion* moments) const
{
    //set the origin
    fOrigin[0] = target_origin[0];
    fOrigin[1] = target_origin[1];
    fOrigin[2] = target_origin[2];

    //construct the wire/triangle/rectangle, and then compute its moments
    if(vertices != NULL && moments != NULL)
    {
        moments->Clear();
        int n_vertices = vertices->GetNPoints();

        if(n_vertices == 1) //we have a point
        {
            //compute the difference between the triangle vertex and the target origin
            for(unsigned int i=0; i<3; i++)
            {
                fDel[i] = (vertices->GetPoint(0))[i] - target_origin[i];
            }

            KFMMath::IrregularSolidHarmonic_Cart_Array(fDegree, fDel, fSolidHarmonics);

            for(unsigned int i=0; i<fSize; i++)
            {
                fMoments[i] = std::conj(fSolidHarmonics[i]);
            }

            moments->SetMoments(&fMoments);

            return false;
        }
        if(n_vertices == 2 ) //we have a wire
        {
            fLine->SetVertices(vertices->GetPoint(0), vertices->GetPoint(1));
            fLine->Initialize();

            double inv_length = 1.0/(fLine->GetLength());
            int psi, nsi;
            double val[2];
            for(int l=0; l <= fDegree; l++)
            {
                fL = l;
                for(int m=0; m <= l; m++)
                {
                    fM = m;
                    psi = l*(l+1) + m;
                    nsi = l*(l+1) - m;
                    fNumInt1D->Integral(val);
                    fMoments[psi] = inv_length*std::complex<double>(val[0], -1*val[1]);
                    fMoments[nsi] = inv_length*std::complex<double>(val[0], val[1]);
                }
            }

            moments->SetMoments(&fMoments);

            return true;
        }
        if(n_vertices == 3) //we have a triangle
        {

            fTriangle->SetVertices(vertices->GetPoint(0), vertices->GetPoint(1), vertices->GetPoint(2));
            fTriangle->Initialize();

            double inv_area = 1.0/(fTriangle->GetArea());

            fNumInt2D->SetSurface(fTriangle);

            int psi, nsi;
            double val[2];
            for(int l=0; l <= fDegree; l++)
            {
                fL = l;
                for(int m=0; m <= l; m++)
                {
                    fM = m;
                    psi = l*(l+1) + m;
                    nsi = l*(l+1) - m;
                    fNumInt2D->Integral(val);
                    fMoments[psi] = inv_area*std::complex<double>(val[0], -1*val[1]);
                    fMoments[nsi] = inv_area*std::complex<double>(val[0], val[1]);
                }
            }

            moments->SetMoments(&fMoments);

            return true;
        }
        if(n_vertices == 4) //we have a rectangle/quadrilateral
        {
            fRectangle->SetVertices(vertices->GetPoint(0), vertices->GetPoint(1), vertices->GetPoint(2), vertices->GetPoint(3));
            fRectangle->Initialize();

            double inv_area = 1.0/(fRectangle->GetArea());

            fNumInt2D->SetSurface(fRectangle);

            int psi, nsi;
            double val[2];
            for(int l=0; l <= fDegree; l++)
            {
                fL = l;
                for(int m=0; m <= l; m++)
                {
                    fM = m;
                    psi = l*(l+1) + m;
                    nsi = l*(l+1) - m;
                    fNumInt2D->Integral(val);
                    fMoments[psi] = inv_area*std::complex<double>(val[0], -1*val[1]);
                    fMoments[nsi] = inv_area*std::complex<double>(val[0], val[1]);
                }
            }

            moments->SetMoments(&fMoments);

            return true;
        }
        else
        {
            kfmout<<"KFMElectrostaticLocalCoefficientCalculatorNumeric::ConstructExpansion: Warning, electrode type not recognized"<<std::endl;
            return false;
        };

    }
    else
    {
        kfmout<<"KFMElectrostaticMultipoleCalculatorAnalytic::ConstructExpansion: Warning, Primitive ID is corrupt or electrode does not exist"<<std::endl;
        return false;
    }
}

void
KFMElectrostaticLocalCoefficientCalculatorNumeric::IrregularSolidHarmonic(const double* point, double* result) const
{
    fDel[0] = point[0] - fOrigin[0];
    fDel[1] = point[1] - fOrigin[1];
    fDel[2] = point[2] - fOrigin[2];

    std::complex<double> temp = KFMMath::IrregularSolidHarmonic_Cart((int)fL, (int)fM, fDel);
    result[0] = std::real(temp);
    result[1] = std::imag(temp);
}



}
