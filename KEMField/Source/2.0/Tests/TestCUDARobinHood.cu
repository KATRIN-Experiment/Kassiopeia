#include <getopt.h>
#include <iostream>
#include <cstdlib>

#include "KSurfaceContainer.hh"
#include "KSurface.hh"
#include "KSurfaceTypes.hh"

#include "KEMConstants.hh"

#include "KElectrostaticBoundaryIntegrator.hh"


#include "KCUDASurfaceContainer.hh"
#include "KCUDAElectrostaticBoundaryIntegrator.hh"
#include "KCUDABoundaryIntegralMatrix.hh"
#include "KCUDABoundaryIntegralVector.hh"
#include "KCUDABoundaryIntegralSolutionVector.hh"
#include "KRobinHood_CUDA.hh"


#include "KCUDAElectrostaticIntegratingFieldSolver.hh"


////////////////////
// VISITOR HEADER //
////////////////////


namespace KEMField{
	template <class Integrator> class KExtractBISolution;

	template <> class KExtractBISolution<KElectrostaticBoundaryIntegrator>
	{
	public:
		typedef KElectrostaticBoundaryIntegrator::Basis theBasis;

		KExtractBISolution( const KSurfaceContainer& theContainer, KElectrostaticBoundaryIntegrator& theIntegrator );
		virtual ~KExtractBISolution() {}

		virtual void Initialize() {}

		double Potential( const KPosition& P ) const;
		KEMThreeVector ElectricField( const KPosition& P ) const;

	protected:
		const KSurfaceContainer& fContainer;
		KElectrostaticBoundaryIntegrator& fIntegrator;

	private:

		class ShapeVisitorForPotential : public KSelectiveVisitor<KShapeVisitor,KElectrostaticBoundaryIntegrator::AcceptedShapes>
		{
		public:
			using KSelectiveVisitor<KShapeVisitor,KElectrostaticBoundaryIntegrator::AcceptedShapes>::Visit;

			ShapeVisitorForPotential( KElectrostaticBoundaryIntegrator& theIntegrator ) : fIntegrator( theIntegrator ) {}

		      void Visit(KTriangle& t) { ComputePotential(t); }
		      void Visit(KRectangle& r) { ComputePotential(r); }
		      void Visit(KLineSegment& l) { ComputePotential(l); }
		      void Visit(KConicSection& c) { ComputePotential(c); }
		      void Visit(KRing& r) { ComputePotential(r); }
		      void Visit(KTriangleGroup& t) { ComputePotential(t); }
		      void Visit(KRectangleGroup& r) { ComputePotential(r); }
		      void Visit(KLineSegmentGroup& l) { ComputePotential(l);}
		      void Visit(KConicSectionGroup& c) { ComputePotential(c); }
		      void Visit(KRingGroup& r) { ComputePotential(r); }

		      template <class ShapePolicy> void ComputePotential( ShapePolicy& s )
		      {
		    	  fPotential = fIntegrator.Potential( &s, fP );
		      }

		      void SetPosition(const KPosition& p) const { fP = p; }
		      double GetNormalizedPotential() const { return fPotential; }

		private:
	      mutable KPosition fP;
	      double fPotential;
	      KElectrostaticBoundaryIntegrator& fIntegrator;
		}; /* ShapeVisitorForPotential */

	    class ShapeVisitorForElectricField : public KSelectiveVisitor<KShapeVisitor, KElectrostaticBoundaryIntegrator::AcceptedShapes>
	    {
	    public:
	      using KSelectiveVisitor<KShapeVisitor,KElectrostaticBoundaryIntegrator::AcceptedShapes>::Visit;

	      ShapeVisitorForElectricField(KElectrostaticBoundaryIntegrator& integrator) : fIntegrator(integrator) {}

	      void Visit(KTriangle& t) { ComputeElectricField(t); }
	      void Visit(KRectangle& r) { ComputeElectricField(r); }
	      void Visit(KLineSegment& l) { ComputeElectricField(l); }
	      void Visit(KConicSection& c) { ComputeElectricField(c); }
	      void Visit(KRing& r) { ComputeElectricField(r); }
	      void Visit(KTriangleGroup& t) { ComputeElectricField(t); }
	      void Visit(KRectangleGroup& r) { ComputeElectricField(r); }
	      void Visit(KLineSegmentGroup& l) { ComputeElectricField(l);}
	      void Visit(KConicSectionGroup& c) { ComputeElectricField(c); }
	      void Visit(KRingGroup& r) { ComputeElectricField(r); }

	      template <class ShapePolicy> void ComputeElectricField(ShapePolicy& s)
	      {
	    	  fElectricField = fIntegrator.ElectricField(&s,fP);
	      }

	      void SetPosition(const KPosition& p) const { fP = p; }
	      KEMThreeVector& GetNormalizedElectricField() const { return fElectricField;}

	    private:
	      mutable KPosition fP;
	      mutable KEMThreeVector fElectricField;
	      KElectrostaticBoundaryIntegrator& fIntegrator;
	    }; /* ShapeVisitorForElectricField */

	    mutable ShapeVisitorForPotential fShapeVisitorForPotential;
	    mutable ShapeVisitorForElectricField fShapeVisitorForElectricField;

	}; /* KExtractBISolution */


///////////////////////
// VISITOR FUNCTIONS //
///////////////////////


	KExtractBISolution<KElectrostaticBoundaryIntegrator>::KExtractBISolution( const KSurfaceContainer& theContainer, KElectrostaticBoundaryIntegrator& theIntegrator ) : fContainer( theContainer ), fIntegrator( theIntegrator ), fShapeVisitorForPotential( theIntegrator ), fShapeVisitorForElectricField( theIntegrator ) {}

	double KExtractBISolution<KElectrostaticBoundaryIntegrator>::Potential( const KPosition& P ) const
	{
		KSurfaceContainer::iterator it;
		double sum( 0. );
		fShapeVisitorForPotential.SetPosition( P );

		for( it=fContainer.begin<theBasis>(); it!=fContainer.end<theBasis>(); ++it ) {
			(*it)->Accept( fShapeVisitorForPotential );
			sum += fShapeVisitorForPotential.GetNormalizedPotential();
		}

		return sum;
	}

  KEMThreeVector KExtractBISolution<KElectrostaticBoundaryIntegrator>::ElectricField(const KPosition& P) const
  {
    fShapeVisitorForElectricField.SetPosition(P);
    KEMThreeVector sum(0.,0.,0.);
    KSurfaceContainer::iterator it;

    for (it=fContainer.begin<theBasis>();it!=fContainer.end<theBasis>();++it) {
      (*it)->Accept(fShapeVisitorForElectricField);
      sum += fShapeVisitorForElectricField.GetNormalizedElectricField();
    }

    return sum;
  }

}

using namespace KEMField;

int main(int /*argc*/, char** /*argv[]*/)
{
  KPosition ori(0.,0.,14.001);

  KSurfaceContainer surfaceContainer;

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tL = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  tL->SetA( 5. ); // positive x-direction
  tL->SetB( 2.5 ); // positive y-direction
  KEMThreeVector tLp0( 0., 0., 8. ); /* P0 */
  tL->SetP0(tLp0);
  KEMThreeVector tLn1( 1., 0., 0. ); /* N1 */
  tL->SetN1( tLn1 );
  KEMThreeVector tLn2( 0., 1., 0. ); /* N2 */
  tL->SetN2( tLn2 );
  //tL->SetSolution(1.); // charge density (electrostatic basis)
  tL->SetBoundaryValue( 1000. ); // electric potential

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* tR = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  tR->SetA( 5. ); // positive x-direction
  tR->SetB( 2.5 ); // positive y-direction
  KEMThreeVector tRp0( 0., 0., 12. ); /* P0 */
  tR->SetP0(tRp0);
  KEMThreeVector tRn1( 1., 0., 0. ); /* N1 */
  tR->SetN1( tRn1 );
  KEMThreeVector tRn2( 0., 1., 0. ); /* N2 */
  tR->SetN2( tRn2 );
  //tL->SetSolution(1.); // charge density (electrostatic basis)
  tR->SetBoundaryValue( 1000. ); // electric potential

  KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>* t3 = new KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle>();
  t3->SetA( 5. ); // positive x-direction
  t3->SetB( 2.5 ); // positive y-direction
  KEMThreeVector t3p0( 0., 0., 14. ); /* P0 */
  t3->SetP0(t3p0);
  KEMThreeVector t3n1( 1., 0., 0. ); /* N1 */
  t3->SetN1( t3n1 );
  KEMThreeVector t3n2( 0., 1., 0. ); /* N2 */
  t3->SetN2( t3n2 );
  //t3->SetSolution(1.); // charge density (electrostatic basis)
  t3->SetBoundaryValue( 1000. ); // electric potential

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KLineSegment>* w = new KSurface<KElectrostaticBasis,
					   KDirichletBoundary,
					   KLineSegment>();

  w->SetP0(KEMThreeVector(-0.457222,0.0504778,-0.51175));
  w->SetP1(KEMThreeVector(-0.463342,0.0511534,-0.515712));
  w->SetDiameter(0.0003);
  w->SetBoundaryValue(-900);

  // left triangle
KPosition t(0.5,0.5,9.);

surfaceContainer.push_back( tL );
surfaceContainer.push_back( tR );
surfaceContainer.push_back( t3 );

    KCUDASurfaceContainer* cudaSurfaceContainer = new KCUDASurfaceContainer(surfaceContainer);
    KCUDAInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );
    

    KCUDAElectrostaticBoundaryIntegrator integrator( *cudaSurfaceContainer );
    KBoundaryIntegralMatrix<KCUDABoundaryIntegrator<KElectrostaticBasis> > A(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralVector<KCUDABoundaryIntegrator<KElectrostaticBasis> > b(*cudaSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KCUDABoundaryIntegrator<KElectrostaticBasis> > x(*cudaSurfaceContainer,integrator);
    
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_CUDA> robinHood;
    
    robinHood.SetTolerance( 1e-6 );
    robinHood.SetResidualCheckInterval( 1 );

    KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >* display = new KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >();
    display->Interval( 1 );
   	robinHood.AddVisitor( display );
                        
    robinHood.Solve(A,x,b);

	KElectrostaticBoundaryIntegrator integr;
	KExtractBISolution<KElectrostaticBoundaryIntegrator> myExtractor(surfaceContainer, integr );
	std::cout << "buffered: " << cudaSurfaceContainer->GetNBufferedElements() << std::endl;
	std::cout << "origin: " << myExtractor.Potential( ori ) << std::endl;

            KCUDAData* data = KCUDAInterface::GetInstance()->GetActiveData();
//            if( data )
//                oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
//            else
//            {
                //KCUDASurfaceContainer* newSurfaceContainer = new KCUDASurfaceContainer( surfaceContainer );
            cudaSurfaceContainer = new KCUDASurfaceContainer( surfaceContainer );
            KCUDAInterface::GetInstance()->SetActiveData( cudaSurfaceContainer );
//            }
            //KCUDAElectrostaticBoundaryIntegrator integrator2( *cudaSurfaceContainer2 );

                KIntegratingFieldSolver< KCUDAElectrostaticBoundaryIntegrator >* fCUDAIntegratingFieldSolver
				= new KIntegratingFieldSolver< KCUDAElectrostaticBoundaryIntegrator >( *cudaSurfaceContainer, integrator );

            //KCUDAElectrostaticBoundaryIntegrator* fCUDAIntegrator = new KOpenCLElectrostaticBoundaryIntegrator( *oclContainer );
			//KIntegratingFieldSolver< KCUDAElectrostaticBoundaryIntegrator >* fCUDAIntegratingFieldSolver
				//= new KIntegratingFieldSolver< KCUDAElectrostaticBoundaryIntegrator >( *cudaSurfaceContainer, integrator );
            fCUDAIntegratingFieldSolver->Initialize();
            //fCUDAIntegratingFieldSolver->ConstructCUDAKernels();
            //fCUDAIntegratingFieldSolver->AssignDeviceMemory();
            std::cout << "origin-int: " << fCUDAIntegratingFieldSolver->Potential( ori ) << std::endl;

//        }



  return 0;
}
