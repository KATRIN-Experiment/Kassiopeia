#include "KGConicSectPortHousingSurfaceMesher.hh"

#include <algorithm>

#include "KGLinearCongruentialGenerator.hh"

#include "KGMeshTriangle.hh"
#include "KGMeshRectangle.hh"

namespace KGeoBag
{
  void KGConicSectPortHousingSurfaceMesher::VisitWrappedSurface( KGConicSectPortHousingSurface* conicSectPortHousingSurface )
  {
    fConicSectPortHousing = conicSectPortHousingSurface->GetObject();

    std::vector< double > z_mid( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > r_mid( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > theta( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > z_length( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > alpha( fConicSectPortHousing->GetNPorts(), 0 );

    ComputeEnclosingBoxDimensions( z_mid, r_mid, theta, z_length, alpha );

    // the ports are complicated, so they compute themselves.
    KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer* paraxialPortDiscretizer = new KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer( this );
    KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer* orthogonalPortDiscretizer = new KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer( this );
    for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
    {
      if( const KGConicSectPortHousing::ParaxialPort* pP = dynamic_cast< const KGConicSectPortHousing::ParaxialPort* >( fConicSectPortHousing->GetPort( i ) ) )
	paraxialPortDiscretizer->DiscretizePort( pP );
      else if( const KGConicSectPortHousing::OrthogonalPort* oP = dynamic_cast< const KGConicSectPortHousing::OrthogonalPort* >( fConicSectPortHousing->GetPort( i ) ) )
	orthogonalPortDiscretizer->DiscretizePort( oP );
    }
    delete paraxialPortDiscretizer;
    delete orthogonalPortDiscretizer;

    // all that is left is to fill in the gaps around the ports.

    // z0: the z-intercept of the generating line for the conic section housing
    double z0 = (fConicSectPortHousing->GetZAMain() - ((fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain())) * fConicSectPortHousing->GetRAMain());

    std::vector< double > phi( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > z_width( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > r_width( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > z_in( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > r_in( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > z_out( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > r_out( fConicSectPortHousing->GetNPorts(), 0 );

    for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
    {
      const KGConicSectPortHousing::Port* p = fConicSectPortHousing->GetPort( i );
      phi[ i ] = p->GetBoxAngle();

      r_width[ i ] = p->GetBoxROuter() - p->GetBoxRInner();
      z_width[ i ] = r_width[ i ] * ((fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()));

      r_in[ i ] = p->GetBoxRInner();
      z_in[ i ] = (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) * r_in[ i ] + z0;

      r_out[ i ] = p->GetBoxROuter();
      z_out[ i ] = (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) * r_out[ i ] + z0;
    }

    // now that we have the parameters of all of the port boxes, we fill in the
    // gaps

    double phiStart = 0.;
    if( fConicSectPortHousing->GetNPorts() > 0 )
      phiStart = theta[ 0 ];

    for( int i = 0; i < fConicSectPortHousing->GetPolyMain(); i++ )
    {
      // midpoint of the rectangle
      double phi1 = phiStart + (i * 2. * M_PI / fConicSectPortHousing->GetPolyMain() + M_PI / fConicSectPortHousing->GetPolyMain());
      double phi2 = phiStart + ((i + 1.) * 2. * M_PI / fConicSectPortHousing->GetPolyMain() + M_PI / fConicSectPortHousing->GetPolyMain());

      std::vector< double > rinterval_start;
      std::vector< double > rinterval_end;
      std::vector< double > zinterval_start;
      std::vector< double > zinterval_end;

      rinterval_start.push_back( fConicSectPortHousing->GetRAMain() );
      zinterval_start.push_back( fConicSectPortHousing->GetZAMain() );
      for( unsigned int j = 0; j < fConicSectPortHousing->GetNPorts(); j++ )
      {
	if( ChordsIntersect( phi1, (phi2 + phi1) * .5, theta[ j ] - phi[ j ] / 2., theta[ j ] + phi[ j ] / 2. ) )
	{
	  rinterval_end.push_back( r_out[ j ] );
	  zinterval_end.push_back( z_out[ j ] );
	  rinterval_start.push_back( r_in[ j ] );
	  zinterval_start.push_back( z_in[ j ] );
	}
      }
      rinterval_end.push_back( fConicSectPortHousing->GetRBMain() );
      zinterval_end.push_back( fConicSectPortHousing->GetZBMain() );

      std::sort( zinterval_start.begin(), zinterval_start.end() );
      std::sort( zinterval_end.begin(), zinterval_end.end() );

      if( fConicSectPortHousing->GetRAMain() < fConicSectPortHousing->GetRBMain() )
      {
	std::sort( rinterval_start.begin(), rinterval_start.end() );
	std::sort( rinterval_end.begin(), rinterval_end.end() );
      }
      else
      {
	std::sort( rinterval_start.rbegin(), rinterval_start.rend() );
	std::sort( rinterval_end.rbegin(), rinterval_end.rend() );
      }

      for( unsigned int j = 0; j < rinterval_start.size(); j++ )
      {
	if( fabs( zinterval_start[ j ] - zinterval_end[ j ] ) < 1.e-10 )
	  continue;

	int nInc = fConicSectPortHousing->GetNumDiscMain() * (zinterval_end[ j ] - zinterval_start[ j ]) / (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain());

	// account for small gaps between ports
	if( nInc == 0 )
	  nInc++;

	for( int m = 0; m < nInc; m++ )
	{
	  double r_min = rinterval_start.at( j ) + (double( m )) / nInc * (rinterval_end.at( j ) - rinterval_start.at( j ));
	  double r_max = rinterval_start.at( j ) + (double( m + 1 )) / nInc * (rinterval_end.at( j ) - rinterval_start.at( j ));
	  double z_min = zinterval_start.at( j ) + (double( m )) / nInc * (zinterval_end.at( j ) - zinterval_start.at( j ));
	  double z_max = zinterval_start.at( j ) + (double( m + 1 )) / nInc * (zinterval_end.at( j ) - zinterval_start.at( j ));
	  double inc1 = 2. * M_PI / fConicSectPortHousing->GetPolyMain();
	  double inc2 = 2 * asin( r_min / r_max * sin( inc1 * .5 ) );

	  KThreeVector P0; // first corner
	  KThreeVector P1; // second corner
	  KThreeVector P2; // third corner
	  KThreeVector P3; // fourth corner

	  P0[ 2 ] = P1[ 2 ] = z_min;
	  P2[ 2 ] = P3[ 2 ] = z_max;

	  P0[ 0 ] = r_min * cos( phi1 - inc1 * .5 );
	  P0[ 1 ] = r_min * sin( phi1 - inc1 * .5 );
	  P1[ 0 ] = r_min * cos( phi1 + inc1 * .5 );
	  P1[ 1 ] = r_min * sin( phi1 + inc1 * .5 );
	  P2[ 0 ] = r_max * cos( phi1 - inc2 * .5 );
	  P2[ 1 ] = r_max * sin( phi1 - inc2 * .5 );
	  P3[ 0 ] = r_max * cos( phi1 + inc2 * .5 );
	  P3[ 1 ] = r_max * sin( phi1 + inc2 * .5 );

	  // lengths of the sides of the rectangle
	  // double s1 = sqrt( (P1[ 0 ] - P0[ 0 ]) * (P1[ 0 ] - P0[ 0 ]) + (P1[ 1 ] - P0[ 1 ]) * (P1[ 1 ] - P0[ 1 ]) + (P1[ 2 ] - P0[ 2 ]) * (P1[ 2 ] - P0[ 2 ]) );
	  // double s2 = sqrt( (P2[ 0 ] - P0[ 0 ]) * (P2[ 0 ] - P0[ 0 ]) + (P2[ 1 ] - P0[ 1 ]) * (P2[ 1 ] - P0[ 1 ]) + (P2[ 2 ] - P0[ 2 ]) * (P2[ 2 ] - P0[ 2 ]) );

	  // double a=0;
	  // double b=0;
	  // double n1[3];
	  // double n2[3];

	  // if (s1>s2)
	  // {
	  //   a = s1;
	  //   b = s2;
	  //   for (int k=0;k<3;k++)
	  //   {
	  //     n1[k] = (P1[k]-P0[k])/s1;
	  //     n2[k] = (P2[k]-P0[k])/s2;
	  //   }
	  // }
	  // else
	  // {
	  //   a = s2;
	  //   b = s1;
	  //   for (int k=0;k<3;k++)
	  //   {
	  //     n1[k] = (P2[k]-P0[k])/s2;
	  //     n2[k] = (P1[k]-P0[k])/s1;
	  //   }
	  // }

	  // the corners of the next two triangles are P1, P3, P4 and P1, P4, P5
	  KThreeVector P4;
	  KThreeVector P5;
	  P5[ 0 ] = r_max * cos( phi2 - inc2 * .5 );
	  P5[ 1 ] = r_max * sin( phi2 - inc2 * .5 );
	  P5[ 2 ] = z_max;
	  P4[ 0 ] = (P5[ 0 ] + P3[ 0 ]) * .5;
	  P4[ 1 ] = (P5[ 1 ] + P3[ 1 ]) * .5;
	  P4[ 2 ] = (P5[ 2 ] + P3[ 2 ]) * .5;

	  KGMeshTriangle* t = new KGMeshTriangle( P0, P1, P5 );
	  AddElement( t );
	  t = new KGMeshTriangle( P0, P2, P5 );
	  AddElement( t );
	}
      }
    }
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::ComputeEnclosingBoxDimensions( std::vector< double >& z_mid, std::vector< double >& r_mid, std::vector< double >& theta, std::vector< double >& z_length, std::vector< double >& alpha )
  {
    // This function computes the lengths of a sides of the boxes that enclose
    // the special discretization around the port holes.  Since we are dealing
    // with a conic section, the boxes are actually fan-shaped openings.

    // z_mid[i]:    the z-coordinate of the middle of box i
    // r_mid[i]:    the r-coordinate of the middle of box i
    // theta[i]:    the angle of the mid-point of box i
    // z_length[i]: the length of box i in z
    // alpha[i]:    the opening angle of box i

    // z0: the z-intercept of the generating line for the conic section housing
    double z0 = (fConicSectPortHousing->GetZAMain() - ((fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain())) * fConicSectPortHousing->GetRAMain());

    for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
    {
      const KGConicSectPortHousing::Port* p = fConicSectPortHousing->GetPort( i );

      if( const KGConicSectPortHousing::ParaxialPort* pP = dynamic_cast< const KGConicSectPortHousing::ParaxialPort* >( p ) )
	r_mid[ i ] = sqrt( pP->GetASub( 0 ) * pP->GetASub( 0 ) + pP->GetASub( 1 ) * pP->GetASub( 1 ) );

      else if( const KGConicSectPortHousing::OrthogonalPort* oP = dynamic_cast< const KGConicSectPortHousing::OrthogonalPort* >( p ) )
	r_mid[ i ] = sqrt( oP->GetCen( 0 ) * oP->GetCen( 0 ) + oP->GetCen( 1 ) * oP->GetCen( 1 ) );

      z_mid[ i ] = (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) * r_mid[ i ] + z0;

      theta[ i ] = fabs( atan( p->GetASub( 1 ) / p->GetASub( 0 ) ) );
      if( p->GetASub( 1 ) > 0 )
      {
	if( p->GetASub( 0 ) < 0 )
	  theta[ i ] = M_PI - theta[ i ];
      }
      else
      {
	if( p->GetASub( 0 ) < 0 )
	  theta[ i ] += M_PI;
	else
	  theta[ i ] = 2. * M_PI - theta[ i ];
      }

      if( const KGConicSectPortHousing::ParaxialPort* pP = dynamic_cast< const KGConicSectPortHousing::ParaxialPort* >( p ) )
	alpha[ i ] = 2 * asin( pP->GetRSub() / fabs( z_mid[ i ] - z0 ) );

      else if( const KGConicSectPortHousing::OrthogonalPort* oP = dynamic_cast< const KGConicSectPortHousing::OrthogonalPort* >( p ) )
	alpha[ i ] = 2 * asin( 2 * oP->GetRSub() );

      double z1 = (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) * (r_mid[ i ] - p->GetRSub()) + z0;
      double z2 = (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) / (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) * (r_mid[ i ] + p->GetRSub()) + z0;

      z_length[ i ] = fabs( z1 - z2 );
    }

    // for more than one port, we find the minimum angle between the ports

    double theta_ref = 0;
    int tmpPoly = fConicSectPortHousing->GetPolyMain();
    double dTheta = 2. * M_PI / tmpPoly;

    if( fConicSectPortHousing->GetNPorts() > 1 )
    {
      double deltaTheta = 2. * M_PI;
      for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
      {
	for( unsigned int j = 0; j < i; j++ )
	{
	  double tmp = fabs( theta[ i ] - theta[ j ] );
	  if( tmp < deltaTheta && tmp > 1.e-3 )
	  {
	    theta_ref = theta[ i ];
	    deltaTheta = tmp;
	  }
	}
      }

      // now that we have the minimum angle between the ports, we find a theta
      // increment that is amenable to all of the ports (there may be a better
      // way to do this)

      dTheta = 0;
      bool dTheta_isGood = true;

      for( int i = 1; i < 1000; i++ )
      {
	dTheta = deltaTheta / i;
	dTheta_isGood = true;

	// first we check that dTheta divides the unit circle evenly...
	if( fmod( 2. * M_PI, dTheta ) > 1.e-3 && fabs( fmod( 2. * M_PI, dTheta ) - dTheta ) > 1.e-3 )
	{
	  dTheta_isGood = false;
	  continue;
	}

	// ...then we check that it divides the angles between all of the ports
	// evenly
	for( unsigned int j = 0; j < fConicSectPortHousing->GetNPorts(); j++ )
	{
	  if( fmod( theta[ j ] - theta_ref, dTheta ) > 1.e-3 &&
	      fabs(fmod( theta[ j ] - theta_ref, dTheta ) - dTheta) > 1.e-3)
	  {
	    dTheta_isGood = false;
	    continue;
	  }
	}

	if( dTheta_isGood )
	  break;
      }

      if( !dTheta_isGood )
      {
	std::stringstream s;
	s << "Unable to find a theta increment that accommodated all of the ports.";
	// KIOManager::GetInstance()->
	//   Message("ConicSectPortHousing","ComputeEnclosingBoxDimensions",s.str(),2);
      }

      // now that we have a minimum theta increment that corresponds to the axes
      // of all of the ports, our main conic section will be discretized according
      // to this value

      tmpPoly = 1;
      int i = 1;
      while( tmpPoly < fConicSectPortHousing->GetPolyMain() )
      {
	tmpPoly = floor( 2. * M_PI / dTheta + .5 ) * i;
	i++;
      }
      fConicSectPortHousing->SetPolyMain( tmpPoly );
    }

    // now that we have properly set the poly parameter for our main cylinder,
    // we compute the enclosing box lengths for the ports

    std::vector< double > alpha_target( fConicSectPortHousing->GetNPorts(), 0 );
    std::vector< double > zlen_target( fConicSectPortHousing->GetNPorts(), 0 );

    for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
    {
      zlen_target[ i ] = 1.5 * z_length[ i ];
      double alpha_len = r_mid[ i ] * alpha[ i ];
      alpha_target[ i ] = zlen_target[ i ] / alpha_len * alpha[ i ];
    }

    // We now have the ideal sizes for the ports' bounding boxes.  Next, we see
    // if they will fit, and modify them if they don't.

    bool recalculate = true;

    while( recalculate )
    {
      recalculate = false;

      for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
      {
	const KGConicSectPortHousing::Port* p = fConicSectPortHousing->GetPort( i );
	const KGConicSectPortHousing::ParaxialPort* pP = dynamic_cast< const KGConicSectPortHousing::ParaxialPort* >( p );
	const KGConicSectPortHousing::OrthogonalPort* oP = dynamic_cast< const KGConicSectPortHousing::OrthogonalPort* >( p );

	double zdist_min = 0; // an absolute min for the bounding box length
	double zdist_max = 1.e30; // an absolute max for the bounding box length
	double alpha_min = 0; // an absolute min for the bounding box length
	double alpha_max = 1.e30; // an absolute max for the bounding box length

	zdist_min = z_length[ i ];
	zdist_max = 2. * (z_mid[ i ] - fConicSectPortHousing->GetZAMain());
	if( 2. * (fConicSectPortHousing->GetZBMain() - z_mid[ i ]) < zdist_max )
	  zdist_max = 2. * (fConicSectPortHousing->GetZBMain() - z_mid[ i ]);

	if( pP )
	  alpha_min = 2 * asin( pP->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	else if( oP )
	  alpha_min = 2 * asin( oP->GetRSub() );
	alpha_max = M_PI;

	if( alpha_target[ i ] < alpha_min )
	  alpha_target[ i ] = alpha_min;
	if( alpha_target[ i ] > alpha_max )
	  alpha_target[ i ] = alpha_max;

	double alpha_act = 0.;

	int nSides = 0;

	// We increment the bounding box length by 2 x the side of the polygon
	// face that comprises the main cylinder.  If it doesn't work, we reduce
	// the target bounding box length and try again.

	while( alpha_act <= alpha_target[ i ] )
	{
	  if( nSides > tmpPoly / 2 )
	  {
	    double tmp_min = 0.;
	    if( pP )
	      tmp_min = 2 * asin( pP->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	    else if( oP )
	      tmp_min = 2 * asin( oP->GetRSub() );
	    alpha_target[ i ] -= tmp_min;
	    alpha_target[ i ] *= .9;
	    alpha_target[ i ] += tmp_min;
	    recalculate = true;
	    break;
	  }

	  nSides += 2;
	  alpha_act = nSides * (2. * M_PI / tmpPoly);
	}

	double zdist_act = zlen_target[ i ];

	// if the calculated bounding box is too small in z, we cap it
	if( zdist_act > zdist_max )
	  zdist_act = zdist_max;
	if( zdist_act < zdist_min )
	  zdist_act = zdist_min;

	// if the calculated bounding box is too small in alpha...
	if( alpha_act < alpha_min )
	{
	  // ...we try increasing our granularity by increasing the number
	  // of sides of the main cylinder polygon
	  tmpPoly += (2. * M_PI / dTheta);
	  recalculate = true;
	  continue;
	}

	// if the calculated bounding box is too large...
	if( alpha_act > alpha_max )
	{
	  //... we reduce our target length.
	  alpha_target[ i ] -= 2 * asin( p->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	  alpha_target[ i ] *= .9;
	  alpha_target[ i ] += 2 * asin( p->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	  recalculate = true;
	  continue;
	}

	double boxTheta = fabs( atan( p->GetASub( 1 ) / p->GetASub( 0 ) ) );
	if( p->GetASub( 1 ) > 0 )
	{
	  if( p->GetASub( 0 ) < 0 )
	    boxTheta = M_PI - boxTheta;
	}
	else
	{
	  if( p->GetASub( 0 ) < 0 )
	    boxTheta += M_PI;
	  else
	    boxTheta = 2. * M_PI - boxTheta;
	}
	p->SetBoxTheta( boxTheta );

	if( pP )
	{
	  // if the calculated bounding box is ok, we set the parameters of the
	  // port to this length
	  pP->SetAlphaPolySub( nSides );

	  pP->SetPolySub( nSides * 4 );
	  pP->SetBoxAngle( alpha_act );
	  pP->SetBoxRInner( ((z_mid[ i ] + zdist_act / 2.) - z0) * (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) / (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) );
	  pP->SetBoxROuter( ((z_mid[ i ] - zdist_act / 2.) - z0) * (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) / (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) );
	  pP->SetXDisc( (nSides > 12 ? nSides : 12) );
	  pP->SetCylDisc( (nSides > 12 ? nSides : 12) );
	}

	else if( oP )
	{
	  // if the calculated bounding box is ok, we set the parameters of the
	  // port to this length
	  oP->SetAlphaPolySub( nSides );

	  oP->SetPolySub( nSides * 4 );
	  oP->SetBoxAngle( alpha_act );
	  oP->SetBoxRInner( ((z_mid[ i ] + zdist_act / 2.) - z0) * (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) / (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) );
	  oP->SetBoxROuter( ((z_mid[ i ] - zdist_act / 2.) - z0) * (fConicSectPortHousing->GetRBMain() - fConicSectPortHousing->GetRAMain()) / (fConicSectPortHousing->GetZBMain() - fConicSectPortHousing->GetZAMain()) );
	  oP->SetXDisc( (nSides > 8 ? nSides : 8) );
	  oP->SetCylDisc( (2 * nSides > 8 ? 2 * nSides : 8) );
	}
      }

      // if the loop gets to this point and no parameters have been changed, we
      // then check for overlaps between bounding boxes.  Otherwise, we start
      // the box length calculation again.

      if( !recalculate )
      {
	// we now have box lengths that satisfy the boundary conditions with
	// the main cylinder.  Next, we make sure they don't overlap

	std::vector< double > phi( fConicSectPortHousing->GetNPorts(), 0. );
	std::vector< double > rmax( fConicSectPortHousing->GetNPorts(), 0. );
	std::vector< double > rmin( fConicSectPortHousing->GetNPorts(), 0. );

	// we start by computing phi
	for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
	{
	  const KGConicSectPortHousing::Port* p = fConicSectPortHousing->GetPort( i );
	  phi[ i ] = p->GetBoxAngle();
	  rmax[ i ] = p->GetBoxROuter();
	  rmin[ i ] = p->GetBoxRInner();
	}

	// now, the check

	for( unsigned int i = 0; i < fConicSectPortHousing->GetNPorts(); i++ )
	{
	  double theta_i_min = theta[ i ] - phi[ i ] / 2.;
	  double theta_i_max = theta[ i ] + phi[ i ] / 2.;
	  double r_i_min = rmin[ i ];
	  double r_i_max = rmax[ i ];

	  for( unsigned int j = 0; j < fConicSectPortHousing->GetNPorts(); j++ )
	  {
	    if( j == i )
	      continue;

	    double theta_j_min = theta[ j ] - phi[ j ] / 2.;
	    double theta_j_max = theta[ j ] + phi[ j ] / 2.;
	    double r_j_min = rmin[ j ];
	    double r_j_max = rmax[ j ];

	    if( ChordsIntersect( theta_i_min, theta_i_max, theta_j_min, theta_j_max ) && LengthsIntersect( r_i_min, r_i_max, r_j_min, r_j_max ) )
	    {
	      const KGConicSectPortHousing::Port* p = fConicSectPortHousing->GetPort( i );

	      alpha_target[ i ] -= 2 * asin( p->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	      alpha_target[ i ] *= .9;
	      alpha_target[ i ] += 2 * asin( p->GetRSub() / fabs( z_mid[ i ] - z0 ) );
	      zlen_target[ i ] -= z_length[ i ];
	      zlen_target[ i ] *= .9;
	      zlen_target[ i ] += z_length[ i ];
	      recalculate = true;
	    }
	  }
	}
      }
    }

    if( tmpPoly != fConicSectPortHousing->GetPolyMain() )
    {
      std::stringstream s;
      s << "In order to properly match boundaries between the valves of this port, the variable ConicSectPortHousing::fPolyMain has been modified from " << fConicSectPortHousing->GetPolyMain() << " to " << tmpPoly << ".";

      // KIOManager::GetInstance()->
      // 	Message("ConicSectPortHousing","ComputeEnclosingBoxLengths",s.str(),0);
      fConicSectPortHousing->SetPolyMain( tmpPoly );
    }
  }

  //______________________________________________________________________________

  bool KGConicSectPortHousingSurfaceMesher::ChordsIntersect( double theta1min, double theta1max, double theta2min, double theta2max )
  {
    // determines if chord 1 with endpoints <theta1min>, <theta1max> intersects
    // chord 2 with endpoints <theta2min>, <theta2max> on the unit circle.

    // first, normalize the angles
    while( theta1min > 2. * M_PI )
      theta1min -= 2. * M_PI;
    while( theta1min < 0 )
      theta1min += 2. * M_PI;
    while( theta1max > 2. * M_PI )
      theta1max -= 2. * M_PI;
    while( theta1max < 0 )
      theta1max += 2. * M_PI;
    while( theta2min > 2. * M_PI )
      theta2min -= 2. * M_PI;
    while( theta2min < 0 )
      theta2min += 2. * M_PI;
    while( theta2max > 2. * M_PI )
      theta2max -= 2. * M_PI;
    while( theta2max < 0 )
      theta2max += 2. * M_PI;

    // if no normalization was needed, this problem's easy:
    if( theta1min < theta1max && theta2min < theta2max )
      return ((theta2min - theta1min > 1.e-5 && theta2min - theta1max < -1.e-5) || (theta2max - theta1min > 1.e-5 && theta2max - theta1max < -1.e-5) || (theta1min - theta2min > 1.e-5 && theta1min - theta2max < -1.e-5) || (theta1max - theta2min > 1.e-5 && theta1max - theta2max < -1.e-5) || ((theta2max + theta2min) * .5 > theta1min && (theta2max + theta2min) * .5 < theta1max));

    // otherwise, we have to shift our frame by performing a cut in a region
    // where we know there is no chord

    bool flip1 = (theta1max < theta1min);
    bool flip2 = (theta2max < theta2min);

    KGLinearCongruentialGenerator pseudoRandomGenerator;

    double random = 0;

    while( true )
    {
      random = 2. * M_PI * pseudoRandomGenerator.Random();

      if( (flip1 && (random > theta1min || random < theta1max)) || (!flip1 && (random > theta1min && random < theta1max)) )
	continue;
      if( (flip2 && (random > theta2min || random < theta2max)) || (!flip2 && (random > theta2min && random < theta2max)) )
	continue;
      break;
    }

    // so <random> is a point on the line from 0 to 2 Pi that does not intersect
    // either chord.  Let's shift it to zero

    theta1min -= random;
    if( theta1min < 0 )
      theta1min += 2. * M_PI;
    theta1max -= random;
    if( theta1max < 0 )
      theta1max += 2. * M_PI;
    theta2min -= random;
    if( theta2min < 0 )
      theta2min += 2. * M_PI;
    theta2max -= random;
    if( theta2max < 0 )
      theta2max += 2. * M_PI;

    return ((theta2min - theta1min > 1.e-5 && theta2min - theta1max < -1.e-5) || (theta2max - theta1min > 1.e-5 && theta2max - theta1max < -1.e-5) || (theta1min - theta2min > 1.e-5 && theta1min - theta2max < -1.e-5) || (theta1max - theta2min > 1.e-5 && theta1max - theta2max < -1.e-5) || ((theta2max + theta2min) * .5 > theta1min && (theta2max + theta2min) * .5 < theta1max));
  }

  //______________________________________________________________________________

  bool KGConicSectPortHousingSurfaceMesher::LengthsIntersect( double x1min, double x1max, double x2min, double x2max )
  {
    // determines if chord 1 with endpoints <theta1min>, <theta1max> intersects
    // chord 2 with endpoints <theta2min>, <theta2max> on the unit circle.

    return ((x2min - x1min > 1.e-5 && x2min - x1max < -1.e-5) || (x2max - x1min > 1.e-5 && x2max - x1max < -1.e-5) || (x1min - x2min > 1.e-5 && x1min - x2max < -1.e-5) || (x1max - x2min > 1.e-5 && x1max - x2max < -1.e-5) || ((x2max + x2min) * .5 > x1min && (x2max + x2min) * .5 < x1max));
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer::DiscretizePort( const KGConicSectPortHousing::ParaxialPort* paraxialPort )
  {
    fParaxialPort = paraxialPort;
    const KGConicSectPortHousing* portHousing = paraxialPort->GetPortHousing();

    // first, we map out points at the intersection of the cylinder and conic
    // section
    std::vector< double > x_int;
    std::vector< double > y_int;
    std::vector< double > z_int;

    double z0 = portHousing->GetZAMain() - ((portHousing->GetZBMain() - portHousing->GetZAMain()) / (portHousing->GetRBMain() - portHousing->GetRAMain())) * portHousing->GetRAMain();
    double theta = 2. * atan( (portHousing->GetRBMain() - portHousing->GetRAMain()) / (portHousing->GetZBMain() - portHousing->GetZAMain()) );

    for( int i = 0; i < paraxialPort->GetPolySub(); i++ )
    {
      double v = (i * (2. * M_PI)) / paraxialPort->GetPolySub();

      x_int.push_back( paraxialPort->GetRSub() * cos( v ) + paraxialPort->GetASub( 0 ) );
      y_int.push_back( paraxialPort->GetRSub() * sin( v ) + paraxialPort->GetASub( 1 ) );
      double w = portHousing->GetZAlongConicSect( sqrt( x_int[ i ] * x_int[ i ] + y_int[ i ] * y_int[ i ] ) );
      z_int.push_back( w );
    }

    // we then stratify the intersection on the surface of the sub cylinder

    // the length along the subordinate cylinder to discretize in compensation for
    // the asymmetries associated with the intersection
    double merge_length = paraxialPort->GetAsymmetricLength();
    if( paraxialPort->GetAsymmetricLength() > paraxialPort->GetSymmetricLength() )
      merge_length = paraxialPort->GetSymmetricLength() - paraxialPort->GetAsymmetricLength() * .5;

    std::vector< double > dz( 2 * paraxialPort->GetXDisc(), 0 );
    std::vector< double > alpha( 2 * paraxialPort->GetXDisc(), 0 );

    // for each of these interval calculations, we compute 2 x the length we are
    // interested in, since the intervals start small, get large, and become
    // small again in a symmetric manner
    DiscretizeInterval( 2 * merge_length, 2 * paraxialPort->GetXDisc(), 2., dz );
    DiscretizeInterval( 2., 2 * paraxialPort->GetXDisc(), 2., alpha );

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for( int i = 1; i < paraxialPort->GetXDisc(); i++ )
    {
      dz[ i ] += dz[ i - 1 ];
      alpha[ i ] += alpha[ i - 1 ];
    }

    // we flip alpha to go from 1 to zero over <paraxialPort->GetPolySub()> increments
    alpha[ 0 ] = 1.;
    for( int i = 1; i < paraxialPort->GetXDisc(); i++ )
      alpha[ i ] = 1. - alpha[ i ];

    std::vector< double > z_low( z_int );
    std::vector< double > z_high( paraxialPort->GetPolySub(), 0 );

    KThreeVector p0;
    KThreeVector p1;
    KThreeVector p2;
    KThreeVector p3;

    for( int i = 0; i < paraxialPort->GetXDisc(); i++ )
    {
      for( int j = 0; j <= paraxialPort->GetPolySub(); j++ )
      {
	if( j < paraxialPort->GetPolySub() )
	{
	  double v = ((j % paraxialPort->GetPolySub()) * (2. * M_PI)) / paraxialPort->GetPolySub();
	  double t = atan( (paraxialPort->GetRSub() * sin( v ) + paraxialPort->GetASub( 1 )) / (paraxialPort->GetRSub() * cos( v ) + paraxialPort->GetASub( 0 )) );
	  double u = (paraxialPort->GetRSub() * cos( v ) + paraxialPort->GetASub( 0 )) / (sin( theta / 2. ) * cos( t ));
	  double w = (paraxialPort->GetRSub() * cos( v ) + paraxialPort->GetASub( 0 )) / fabs( paraxialPort->GetRSub() * cos( v ) + paraxialPort->GetASub( 0 ) ) * u * cos( theta / 2. ) + z0;

	  z_high[ j ] = dz[ i ] + w * alpha[ i ];
	}

	if( j != 0 )
	{
	  p0[ 0 ] = p1[ 0 ] = x_int[ (j - 1) % paraxialPort->GetPolySub() ];
	  p0[ 1 ] = p1[ 1 ] = y_int[ (j - 1) % paraxialPort->GetPolySub() ];
	  p0[ 2 ] = z_high[ (j - 1) % paraxialPort->GetPolySub() ];
	  p1[ 2 ] = z_low[ (j - 1) % paraxialPort->GetPolySub() ];

	  p2[ 0 ] = p3[ 0 ] = x_int[ j % paraxialPort->GetPolySub() ];
	  p2[ 1 ] = p3[ 1 ] = y_int[ j % paraxialPort->GetPolySub() ];
	  p2[ 2 ] = z_high[ j % paraxialPort->GetPolySub() ];
	  p3[ 2 ] = z_low[ j % paraxialPort->GetPolySub() ];

	  // now, we cast the points into triangle-form
	  KGMeshTriangle* t;
	  t = new KGMeshTriangle( p0, p1, p2 );
	  fConicSectPortHousingDiscretizer->AddElement( t );
	  t = new KGMeshTriangle( p3, p1, p2 );
	  fConicSectPortHousingDiscretizer->AddElement( t );
	}
      }
      for( int j = 0; j < paraxialPort->GetPolySub(); j++ )
	z_low[ j ] = z_high[ j ];
    }

    // now that the neck of the subordinate cylinder is discretized, we start on
    // the area on the main conic section surrounding the hole
    paraxialPort->SetXDisc( paraxialPort->GetXDisc() / 2 );
    DiscretizeInterval( 2. * (paraxialPort->GetBoxROuter() - paraxialPort->GetBoxRInner() - paraxialPort->GetRSub()), 2 * paraxialPort->GetXDisc(), 2., dz );

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for( int i = 1; i < paraxialPort->GetXDisc(); i++ )
      dz[ i ] += dz[ i - 1 ];

    double r1, r2;

    std::vector< double > theta1( paraxialPort->GetPolySub(), 0 );
    std::vector< double > theta2( paraxialPort->GetPolySub(), 0 );

    // initialize theta1 array
    for( int j = 0; j < paraxialPort->GetPolySub(); j++ )
      theta1[ j ] = 2. * M_PI * ((double) j) / paraxialPort->GetPolySub();

    for( int i = 0; i < paraxialPort->GetXDisc(); i++ )
    {
      if( i == 0 )
	r1 = paraxialPort->GetRSub();
      else
	r1 = paraxialPort->GetRSub() + dz[ i - 1 ];
      r2 = paraxialPort->GetRSub() + dz[ i ];

      for( int j = 1; j <= paraxialPort->GetPolySub(); j++ )
      {
	// mixed circle-square pattern
	Transition_coord( (j - 1) % paraxialPort->GetPolySub(), r2, p0 );
	Transition_coord( (j - 1) % paraxialPort->GetPolySub(), r1, p1 );
	Transition_coord( j % paraxialPort->GetPolySub(), r2, p2 );
	Transition_coord( j % paraxialPort->GetPolySub(), r1, p3 );

	// to minimize the lengths of the triangles, we swap coordinates if
	// necessary
	double tmp1 = sqrt( (p0[ 0 ] - p3[ 0 ]) * (p0[ 0 ] - p3[ 0 ]) + (p0[ 1 ] - p3[ 1 ]) * (p0[ 1 ] - p3[ 1 ]) + (p0[ 2 ] - p3[ 2 ]) * (p0[ 2 ] - p3[ 2 ]) );

	double tmp2 = sqrt( (p1[ 0 ] - p2[ 0 ]) * (p1[ 0 ] - p2[ 0 ]) + (p1[ 1 ] - p2[ 1 ]) * (p1[ 1 ] - p2[ 1 ]) + (p1[ 2 ] - p2[ 2 ]) * (p1[ 2 ] - p2[ 2 ]) );
	if( tmp2 > tmp1 )
	{
	  double tmp3, tmp4;
	  for( int k = 0; k < 3; k++ )
	  {
	    tmp3 = p0[ k ];
	    tmp4 = p1[ k ];
	    p0[ k ] = p2[ k ];
	    p1[ k ] = p3[ k ];
	    p2[ k ] = tmp3;
	    p3[ k ] = tmp4;
	  }
	}

	// now, we cast the global points into triangle-form
	KGMeshTriangle* t;
	t = new KGMeshTriangle( p0, p1, p2 );
	fConicSectPortHousingDiscretizer->AddElement( t );
	t = new KGMeshTriangle( p1, p2, p3 );
	fConicSectPortHousingDiscretizer->AddElement( t );
      }
      for( int k = 0; k < paraxialPort->GetPolySub(); k++ )
	theta1[ k ] = theta2[ k ];
    }

    // now that the tricky part's over, we have only to finish discretizing the
    // cylindrical portion of the valve

    if( paraxialPort->GetAsymmetricLength() < paraxialPort->GetSymmetricLength() )
    {
      dz.empty();
      dz.resize( 2 * paraxialPort->GetCylDisc() );
      DiscretizeInterval( 2. * (paraxialPort->GetSymmetricLength() - paraxialPort->GetAsymmetricLength() / 2.), 2 * paraxialPort->GetCylDisc(), 1.25, dz );
 
      double z = z_high[ 0 ];
      double z_last = z_high[ 0 ];

      double a, b;
      KThreeVector n1, n2;

      a = paraxialPort->GetRSub() * sqrt( 2. * (1. - cos( 2. * M_PI / paraxialPort->GetPolySub() )) );

      for( int i = 0; i < paraxialPort->GetCylDisc(); i++ )
      {
	z_last = z;
	z += dz[ i ];

	b = dz[ i ];

	for( int j = 1; j <= paraxialPort->GetPolySub(); j++ )
	{
	  Circle_coord( (j - 1) % paraxialPort->GetPolySub(), paraxialPort->GetRSub(), p0 );
	  p0[ 2 ] = z_last;

	  Circle_coord( j % paraxialPort->GetPolySub(), paraxialPort->GetRSub(), p1 );
	  p1[ 2 ] = z_last;

	  Circle_coord( (j - 1) % paraxialPort->GetPolySub(), paraxialPort->GetRSub(), p2 );
	  p2[ 2 ] = z;

	  double n1mag = 0;
	  double n2mag = 0;

	  for( int k = 0; k < 3; k++ )
	  {
	    n1[ k ] = p1[ k ] - p0[ k ];
	    n1mag += n1[ k ] * n1[ k ];
	    n2[ k ] = p2[ k ] - p0[ k ];
	    n2mag += n2[ k ] * n2[ k ];
	  }

	  n1mag = sqrt( n1mag );
	  n2mag = sqrt( n2mag );

	  for( int k = 0; k < 3; k++ )
	  {
	    n1[ k ] /= n1mag;
	    n2[ k ] /= n2mag;
	  }

	  KGMeshRectangle* r = new KGMeshRectangle( a, b, p0, n1, n2 );
	  fConicSectPortHousingDiscretizer->AddElement( r );
	}
      }
    }
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer::Circle_coord( int i, double /*r*/, double p[ 3 ] )
  {
    // this function returns a point on a circle with radius <r> at angle <theta>

    double theta = 2. * M_PI * ((double) i) / fParaxialPort->GetPolySub() + 7. * M_PI / 4. - fParaxialPort->GetBoxTheta();
    double r_ret = fParaxialPort->GetRSub();

    p[ 0 ] = r_ret * sin( theta ) + fParaxialPort->GetASub( 0 );
    p[ 1 ] = r_ret * cos( theta ) + fParaxialPort->GetASub( 1 );
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer::Fan_coord( int i, double /*r*/, double p[ 3 ] )
  {
    // this function returns a point on a fan inscribing a circle with
    // radius <r> at angle <theta>

    int radDisc = (fParaxialPort->GetPolySub() - 2 * fParaxialPort->GetAlphaPolySub()) / 2;

    if( i < radDisc )
    {
      double r_ret = fParaxialPort->GetBoxRInner() + (double( i )) / radDisc * (fParaxialPort->GetBoxROuter() - fParaxialPort->GetBoxRInner());
      p[ 0 ] = r_ret * cos( fParaxialPort->GetBoxTheta() + fParaxialPort->GetBoxAngle() / 2. );
      p[ 1 ] = r_ret * sin( fParaxialPort->GetBoxTheta() + fParaxialPort->GetBoxAngle() / 2. );
    }
    else if( i < (radDisc + fParaxialPort->GetAlphaPolySub()) )
    {
      int j = i - radDisc;
      double r_ret = fParaxialPort->GetBoxROuter();
      p[ 0 ] = r_ret * cos( fParaxialPort->GetBoxTheta() + fParaxialPort->GetBoxAngle() / 2. - (double( j )) * fParaxialPort->GetBoxAngle() / fParaxialPort->GetAlphaPolySub() );
      p[ 1 ] = r_ret * sin( fParaxialPort->GetBoxTheta() + fParaxialPort->GetBoxAngle() / 2. - (double( j )) * fParaxialPort->GetBoxAngle() / fParaxialPort->GetAlphaPolySub() );
    }
    else if( i < (2 * radDisc + fParaxialPort->GetAlphaPolySub()) )
    {
      int j = i - (radDisc + fParaxialPort->GetAlphaPolySub());
      double r_ret = fParaxialPort->GetBoxROuter() - (double( j )) / radDisc * (fParaxialPort->GetBoxROuter() - fParaxialPort->GetBoxRInner());
      p[ 0 ] = r_ret * cos( fParaxialPort->GetBoxTheta() - fParaxialPort->GetBoxAngle() / 2. );
      p[ 1 ] = r_ret * sin( fParaxialPort->GetBoxTheta() - fParaxialPort->GetBoxAngle() / 2. );
    }
    else
    {
      int j = i - (2 * radDisc + fParaxialPort->GetAlphaPolySub());
      p[ 0 ] = fParaxialPort->GetBoxRInner() * cos( fParaxialPort->GetBoxTheta() - fParaxialPort->GetBoxAngle() / 2. + (double( j )) * fParaxialPort->GetBoxAngle() / fParaxialPort->GetAlphaPolySub() );
      p[ 1 ] = fParaxialPort->GetBoxRInner() * sin( fParaxialPort->GetBoxTheta() - fParaxialPort->GetBoxAngle() / 2. + (double( j )) * fParaxialPort->GetBoxAngle() / fParaxialPort->GetAlphaPolySub() );
    }
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::ParaxialPortDiscretizer::Transition_coord( int i, double r, double p[ 3 ] )
  {
    // this function combines Circle_coord and Fan_coord to return a point on the
    // main conic section that is more circle-like close to the subordinate
    // cylinder, and more fan-like farther away.

    // we let the path be a pure circle at the intersection, and a pure fan a
    // distance fRSub away from the intersection.  It is therefore assumed that
    // <r> varies from fRSub to fBoxLength/2

    // 0 at the intersection, 1 at the fan's edge
    double ratio = (r - fParaxialPort->GetRSub()) / (fParaxialPort->GetBoxROuter() - fParaxialPort->GetBoxRInner() - fParaxialPort->GetRSub());
    double p_circ[ 3 ];
    double p_fan[ 3 ];

    Circle_coord( i, r, p_circ );
    Fan_coord( i, r, p_fan );

    p[ 0 ] = (1. - ratio) * p_circ[ 0 ] + ratio * p_fan[ 0 ];
    p[ 1 ] = (1. - ratio) * p_circ[ 1 ] + ratio * p_fan[ 1 ];
    p[ 2 ] = fParaxialPort->GetPortHousing()->GetZAlongConicSect( sqrt( p[ 0 ] * p[ 0 ] + p[ 1 ] * p[ 1 ] ) );
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer::DiscretizePort( const KGConicSectPortHousing::OrthogonalPort* orthogonalPort )
  {
    fOrthogonalPort = orthogonalPort;
    const KGConicSectPortHousing* portHousing = orthogonalPort->GetPortHousing();

    // we map out points at the edge of the cylinder in local coordinates
    std::vector< double > x_edge_loc;
    std::vector< double > y_edge_loc;
    std::vector< double > z_edge_loc;

    double theta_edge;
    for( int i = 0; i < orthogonalPort->GetPolySub(); i++ )
    {
      // we use x -> sin and y -> cos, and the angular offset of 5/4 pi to match
      // the mapping in the algorithm ConicSectOrthogonalPort::Fan_coord()
      theta_edge = 2. * M_PI * (((double) i) / orthogonalPort->GetPolySub()) + 5. * M_PI / 4.;

      x_edge_loc.push_back( orthogonalPort->GetRSub() * sin( theta_edge ) );
      y_edge_loc.push_back( orthogonalPort->GetRSub() * cos( theta_edge ) );
      z_edge_loc.push_back( sqrt( (orthogonalPort->GetCen( 0 ) - orthogonalPort->GetASub( 0 )) * (orthogonalPort->GetCen( 0 ) - orthogonalPort->GetASub( 0 )) + (orthogonalPort->GetCen( 1 ) - orthogonalPort->GetASub( 1 )) * (orthogonalPort->GetCen( 1 ) - orthogonalPort->GetASub( 1 )) + (orthogonalPort->GetCen( 2 ) - orthogonalPort->GetASub( 2 )) * (orthogonalPort->GetCen( 2 ) - orthogonalPort->GetASub( 2 )) ) );
    }

    // we then translate these points into the global frame
    std::vector< double > x_edge;
    std::vector< double > y_edge;
    std::vector< double > z_edge;

    for( unsigned int i = 0; i < x_edge_loc.size(); i++ )
    {
      double tmp_loc[ 3 ] =
	{ x_edge_loc[ i ], y_edge_loc[ i ], z_edge_loc[ i ] };
      double tmp[ 3 ];

      orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( tmp_loc, tmp, false );

      std::vector< double > p1( 3, 0 );
      std::vector< double > s1( 3, 0 );
      std::vector< double > s2( 3, 0 );

      for( unsigned int j = 0; j < 3; j++ )
      {
	p1[ j ] = tmp[ j ];
	s1[ j ] = orthogonalPort->GetCen( j );
	s2[ j ] = orthogonalPort->GetASub( j );
      }

      x_edge.push_back( tmp[ 0 ] );
      y_edge.push_back( tmp[ 1 ] );
      z_edge.push_back( tmp[ 2 ] );
    }

    // next, we map out points at the intersection of the cylinder and the conic
    // section by casting them down to the intersection
    std::vector< double > x_int;
    std::vector< double > y_int;
    std::vector< double > z_int;

    std::vector< double > p_edge( 3, 0. );
    std::vector< double > p_int( 3, 0. );
    std::vector< double > norm( 3 );
    for( int i = 0; i < 3; i++ )
      norm[ i ] = -orthogonalPort->GetZ_loc( i );

    for( int i = 0; i < orthogonalPort->GetPolySub(); i++ )
    {
      p_edge[ 0 ] = x_edge[ i ];
      p_edge[ 1 ] = y_edge[ i ];
      p_edge[ 2 ] = z_edge[ i ];

      portHousing->RayConicSectIntersection( p_edge, norm, p_int );

      x_int.push_back( p_int[ 0 ] );
      y_int.push_back( p_int[ 1 ] );
      z_int.push_back( p_int[ 2 ] );

      std::vector< double > p1( 3, 0 );
      std::vector< double > s1( 3, 0 );
      std::vector< double > s2( 3, 0 );

      for( unsigned int j = 0; j < 3; j++ )
      {
	p1[ j ] = p_int[ j ];
	s1[ j ] = orthogonalPort->GetCen( j );
	s2[ j ] = orthogonalPort->GetASub( j );
      }
    }

    // we will need the local coordinates for the intersection later, so we
    // compute them now
    std::vector< double > x_int_loc;
    std::vector< double > y_int_loc;
    std::vector< double > z_int_loc;

    for( unsigned int i = 0; i < x_int.size(); i++ )
    {
      double tmp[ 3 ] =
	{ x_int[ i ], y_int[ i ], z_int[ i ] };
      double tmp_loc[ 3 ];

      orthogonalPort->GetCoordinateTransform()->ConvertToLocalCoords( tmp, tmp_loc, false );

      std::vector< double > p1( 3, 0 );
      std::vector< double > s1( 3, 0 );
      std::vector< double > s2( 3, 0 );

      for( unsigned int j = 0; j < 3; j++ )
	p1[ j ] = tmp_loc[ j ];
      s2[ 2 ] = 1.;

      x_int_loc.push_back( tmp_loc[ 0 ] );
      y_int_loc.push_back( tmp_loc[ 1 ] );
      z_int_loc.push_back( tmp_loc[ 2 ] );
    }

    // we then stratify the intersection on the surface of the sub cylinder

    // the length along the subordinate cylinder to discretize in compensation
    // for the asymmetries associated with the intersection
    double min = 1.e6;
    double max = 0.;

    for( unsigned int i = 0; i < x_int.size(); i++ )
    {
      double cyl_dist = sqrt( (x_edge[ i ] - x_int[ i ]) * (x_edge[ i ] - x_int[ i ]) + (y_edge[ i ] - y_int[ i ]) * (y_edge[ i ] - y_int[ i ]) + (z_edge[ i ] - z_int[ i ]) * (z_edge[ i ] - z_int[ i ]) );

      if( cyl_dist < min )
	min = cyl_dist;
      if( cyl_dist > max )
	max = cyl_dist;
    }

    // often, the difference in heights on the intersection is very small, and
    // this small length creates too narrow triangles.  We can compensate for
    // this by requiring that the merge length be at least 1/5 the length of the
    // cylinder
    double merge_length = 2. * (max - min);
    if( merge_length < .2 * min )
      merge_length = .2 * min;

    std::vector< double > dz( 2 * orthogonalPort->GetXDisc(), 0 );
    std::vector< double > alpha( 2 * orthogonalPort->GetXDisc(), 0 );

    // for each of these interval calculations, we compute 2 x the length we are
    // interested in, since the intervals start small, get large, and become
    // small again in a symmetric manner
    DiscretizeInterval( 2 * merge_length, 2 * orthogonalPort->GetXDisc(), 2., dz );
    DiscretizeInterval( 2., 2 * orthogonalPort->GetXDisc(), 2., alpha );

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for( int i = 1; i < orthogonalPort->GetXDisc(); i++ )
    {
      dz[ i ] += dz[ i - 1 ];
      alpha[ i ] += alpha[ i - 1 ];
    }

    // we flip alpha to go from 1 to zero over <orthogonalPort->GetPolySub()> increments
    alpha[ 0 ] = 1.;
    for( int i = 1; i < orthogonalPort->GetXDisc(); i++ )
      alpha[ i ] = 1. - alpha[ i ];

    std::vector< double > z_low( z_int_loc );
    std::vector< double > z_high( orthogonalPort->GetPolySub(), 0 );

    KThreeVector p0_loc;
    KThreeVector p0;
    KThreeVector p1_loc;
    KThreeVector p1;
    KThreeVector p2_loc;
    KThreeVector p2;
    KThreeVector p3_loc;
    KThreeVector p3;

    // here we perform the discretization of the neck of the intersection

    for( int i = 0; i < orthogonalPort->GetXDisc(); i++ )
    {
      for( int j = 0; j <= orthogonalPort->GetPolySub(); j++ )
      {
	if( j < orthogonalPort->GetPolySub() )
	  z_high[ j ] = dz[ i ] - fabs( z_int_loc[ j ] ) * alpha[ i ];

	if( j != 0 )
	{
	  p0_loc[ 0 ] = p1_loc[ 0 ] = x_int_loc[ (j - 1) % orthogonalPort->GetPolySub() ];
	  p0_loc[ 1 ] = p1_loc[ 1 ] = y_int_loc[ (j - 1) % orthogonalPort->GetPolySub() ];
	  p0_loc[ 2 ] = z_high[ (j - 1) % orthogonalPort->GetPolySub() ];
	  p1_loc[ 2 ] = z_low[ (j - 1) % orthogonalPort->GetPolySub() ];

	  p2_loc[ 0 ] = p3_loc[ 0 ] = x_int_loc[ j % orthogonalPort->GetPolySub() ];
	  p2_loc[ 1 ] = p3_loc[ 1 ] = y_int_loc[ j % orthogonalPort->GetPolySub() ];
	  p2_loc[ 2 ] = z_high[ j % orthogonalPort->GetPolySub() ];
	  p3_loc[ 2 ] = z_low[ j % orthogonalPort->GetPolySub() ];

	  // cast these points into the global frame
	  orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( p0_loc, p0, false );
	  orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( p1_loc, p1, false );
	  orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( p2_loc, p2, false );
	  orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( p3_loc, p3, false );

	  // now, we cast the points into triangle-form
	  KGMeshTriangle* t;
	  t = new KGMeshTriangle( p0, p1, p2 );
	  fConicSectPortHousingDiscretizer->AddElement( t );
	  t = new KGMeshTriangle( p3, p1, p2 );
	  fConicSectPortHousingDiscretizer->AddElement( t );
	}
      }
      for( int j = 0; j < orthogonalPort->GetPolySub(); j++ )
	z_low[ j ] = z_high[ j ];
    }

    // now that the neck of the subordinate cylinder is discretized, we start on
    // the area on the main conic section surrounding the hole

    DiscretizeInterval( 2. * (orthogonalPort->GetBoxROuter() - orthogonalPort->GetBoxRInner() - orthogonalPort->GetRSub()), 2 * orthogonalPort->GetXDisc(), 2., dz );

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for( int i = 1; i < orthogonalPort->GetXDisc(); i++ )
      dz[ i ] += dz[ i - 1 ];

    double r1, r2;

    std::vector< double > theta1( orthogonalPort->GetPolySub(), 0 );
    std::vector< double > theta2( orthogonalPort->GetPolySub(), 0 );

    // initialize theta1 array
    for( int j = 0; j < orthogonalPort->GetPolySub(); j++ )
      theta1[ j ] = 2. * M_PI * ((double) j) / orthogonalPort->GetPolySub();

    for( int i = 0; i < orthogonalPort->GetXDisc(); i++ )
    {
      if( i == 0 )
	r1 = orthogonalPort->GetRSub();
      else
	r1 = orthogonalPort->GetRSub() + dz[ i - 1 ];
      r2 = orthogonalPort->GetRSub() + dz[ i ];

      for( int j = 1; j <= orthogonalPort->GetPolySub(); j++ )
      {

	// mixed circle-square pattern
	Transition_coord( j % orthogonalPort->GetPolySub(), r1, p3, x_int, y_int, z_int );
	Transition_coord( (j - 1) % orthogonalPort->GetPolySub(), r1, p1, x_int, y_int, z_int );
	Transition_coord( (j - 1) % orthogonalPort->GetPolySub(), r2, p0, x_int, y_int, z_int );
	Transition_coord( j % orthogonalPort->GetPolySub(), r2, p2, x_int, y_int, z_int );

	// to minimize the lengths of the triangles, we swap coordinates if
	// necessary
	double tmp1 = sqrt( (p0[ 0 ] - p3[ 0 ]) * (p0[ 0 ] - p3[ 0 ]) + (p0[ 1 ] - p3[ 1 ]) * (p0[ 1 ] - p3[ 1 ]) + (p0[ 2 ] - p3[ 2 ]) * (p0[ 2 ] - p3[ 2 ]) );

	double tmp2 = sqrt( (p1[ 0 ] - p2[ 0 ]) * (p1[ 0 ] - p2[ 0 ]) + (p1[ 1 ] - p2[ 1 ]) * (p1[ 1 ] - p2[ 1 ]) + (p1[ 2 ] - p2[ 2 ]) * (p1[ 2 ] - p2[ 2 ]) );
	if( tmp2 > tmp1 )
	{
	  double tmp3, tmp4;
	  for( int k = 0; k < 3; k++ )
	  {
	    tmp3 = p0[ k ];
	    tmp4 = p1[ k ];
	    p0[ k ] = p2[ k ];
	    p1[ k ] = p3[ k ];
	    p2[ k ] = tmp3;
	    p3[ k ] = tmp4;
	  }
	}

	// now, we cast the global points into triangle-form
	KGMeshTriangle* t;
	t = new KGMeshTriangle( p0, p1, p2 );
	fConicSectPortHousingDiscretizer->AddElement( t );
	t = new KGMeshTriangle( p1, p2, p3 );
	fConicSectPortHousingDiscretizer->AddElement( t );
      }
      for( int k = 0; k < orthogonalPort->GetPolySub(); k++ )
	theta1[ k ] = theta2[ k ];
    }

    // now that the tricky part's over, we have only to finish discretizing the
    // cylindrical portion of the valve

    dz.empty();
    dz.resize( 2 * orthogonalPort->GetCylDisc() );
    DiscretizeInterval( 2. * (orthogonalPort->GetLength() - merge_length), 2 * orthogonalPort->GetCylDisc(), 1.25, dz );

//    double theta = 0;

    double z = z_high[ 0 ];
    double z_last = z_high[ 0 ];

    KThreeVector n1_loc;
    KThreeVector n2_loc;
    KThreeVector n1;
    KThreeVector n2;
    double a, b;

    a = orthogonalPort->GetRSub() * sqrt( 2. * (1. - cos( 2. * M_PI / orthogonalPort->GetPolySub() )) );

    for( int i = 0; i < orthogonalPort->GetCylDisc(); i++ )
    {
      z_last = z;
      z += dz[ i ];

      b = dz[ i ];

      for( int j = 1; j <= orthogonalPort->GetPolySub(); j++ )
      {
//	theta = fmod( orthogonalPort->GetBoxTheta() + 2. * M_PI * ((double) i) / orthogonalPort->GetPolySub(), 2. * M_PI );

	Circle_coord( (j - 1) % orthogonalPort->GetPolySub(), orthogonalPort->GetRSub(), p0_loc, x_int_loc, y_int_loc, z_int_loc );
	p0_loc[ 2 ] = z_last;

	Circle_coord( j % orthogonalPort->GetPolySub(), orthogonalPort->GetRSub(), p1, x_int_loc, y_int_loc, z_int_loc );
	p1[ 2 ] = z_last;

	Circle_coord( (j - 1) % orthogonalPort->GetPolySub(), orthogonalPort->GetRSub(), p2, x_int_loc, y_int_loc, z_int_loc );
	p2[ 2 ] = z;

	double n1mag = 0;
	double n2mag = 0;

	for( int k = 0; k < 3; k++ )
	{
	  n1_loc[ k ] = p1[ k ] - p0_loc[ k ];
	  n1mag += n1_loc[ k ] * n1_loc[ k ];
	  n2_loc[ k ] = p2[ k ] - p0_loc[ k ];
	  n2mag += n2_loc[ k ] * n2_loc[ k ];
	}

	n1mag = sqrt( n1mag );
	n2mag = sqrt( n2mag );

	for( int k = 0; k < 3; k++ )
	{
	  n1_loc[ k ] /= n1mag;
	  n2_loc[ k ] /= n2mag;
	}

	orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( p0_loc, p0, false );
	orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( n1_loc, n1, true );
	orthogonalPort->GetCoordinateTransform()->ConvertToGlobalCoords( n2_loc, n2, true );

	KGMeshRectangle* r = new KGMeshRectangle( a, b, p0, n1, n2 );
	fConicSectPortHousingDiscretizer->AddElement( r );
      }
    }
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer::Circle_coord( int i, double /*r*/, double p[ 3 ], std::vector< double >& x_int, std::vector< double >& y_int, std::vector< double >& z_int )
  {
    // This function returns a point on a circle with radius <r> at angle <theta>.

    // rather than compute these values each time the function is called, it is
    // easier (and makes for a more continuous mesh) to simply use the values that
    // were computed earlier

    p[ 0 ] = x_int[ i ];
    p[ 1 ] = y_int[ i ];
    p[ 2 ] = z_int[ i ];
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer::Fan_coord( int i, double /*r*/, double p[ 3 ] )
  {
    // this function returns a point on a fan inscribing a circle with
    // radius <r> at angle <theta>

    int radDisc = (fOrthogonalPort->GetPolySub() - 2 * fOrthogonalPort->GetAlphaPolySub()) / 2;

    if( i < radDisc )
    {
      double r_ret = fOrthogonalPort->GetBoxRInner() + (double( i )) / radDisc * (fOrthogonalPort->GetBoxROuter() - fOrthogonalPort->GetBoxRInner());
      p[ 0 ] = r_ret * cos( fOrthogonalPort->GetBoxTheta() + fOrthogonalPort->GetBoxAngle() / 2. );
      p[ 1 ] = r_ret * sin( fOrthogonalPort->GetBoxTheta() + fOrthogonalPort->GetBoxAngle() / 2. );
    }
    else if( i < (radDisc + fOrthogonalPort->GetAlphaPolySub()) )
    {
      int j = i - radDisc;
      double r_ret = fOrthogonalPort->GetBoxROuter();
      p[ 0 ] = r_ret * cos( fOrthogonalPort->GetBoxTheta() + fOrthogonalPort->GetBoxAngle() / 2. - (double( j )) * fOrthogonalPort->GetBoxAngle() / fOrthogonalPort->GetAlphaPolySub() );
      p[ 1 ] = r_ret * sin( fOrthogonalPort->GetBoxTheta() + fOrthogonalPort->GetBoxAngle() / 2. - (double( j )) * fOrthogonalPort->GetBoxAngle() / fOrthogonalPort->GetAlphaPolySub() );
    }
    else if( i < (2 * radDisc + fOrthogonalPort->GetAlphaPolySub()) )
    {
      int j = i - (radDisc + fOrthogonalPort->GetAlphaPolySub());
      double r_ret = fOrthogonalPort->GetBoxROuter() - (double( j )) / radDisc * (fOrthogonalPort->GetBoxROuter() - fOrthogonalPort->GetBoxRInner());
      p[ 0 ] = r_ret * cos( fOrthogonalPort->GetBoxTheta() - fOrthogonalPort->GetBoxAngle() / 2. );
      p[ 1 ] = r_ret * sin( fOrthogonalPort->GetBoxTheta() - fOrthogonalPort->GetBoxAngle() / 2. );
    }
    else
    {
      int j = i - (2 * radDisc + fOrthogonalPort->GetAlphaPolySub());
      p[ 0 ] = fOrthogonalPort->GetBoxRInner() * cos( fOrthogonalPort->GetBoxTheta() - fOrthogonalPort->GetBoxAngle() / 2. + (double( j )) * fOrthogonalPort->GetBoxAngle() / fOrthogonalPort->GetAlphaPolySub() );
      p[ 1 ] = fOrthogonalPort->GetBoxRInner() * sin( fOrthogonalPort->GetBoxTheta() - fOrthogonalPort->GetBoxAngle() / 2. + (double( j )) * fOrthogonalPort->GetBoxAngle() / fOrthogonalPort->GetAlphaPolySub() );
    }
  }

  //______________________________________________________________________________

  void KGConicSectPortHousingSurfaceMesher::OrthogonalPortDiscretizer::Transition_coord( int i, double r, double p[ 3 ], std::vector< double >& x_int, std::vector< double >& y_int, std::vector< double >& z_int )
  {
    // this function combines Circle_coord and Fan_coord to return a point on the
    // main conic section that is more circle-like close to the subordinate
    // cylinder, and more fan-like farther away.

    // we let the path be a pure circle at the intersection, and a pure fan a
    // distance fRSub away from the intersection.  It is therefore assumed that
    // <r> varies from fRSub to fBoxLength/2

    // 0 at the intersection, 1 at the fan's edge
    double ratio = (r - fOrthogonalPort->GetRSub()) / (fOrthogonalPort->GetBoxROuter() - fOrthogonalPort->GetBoxRInner() - fOrthogonalPort->GetRSub());

    double p_circ[ 3 ];
    double p_fan[ 3 ];

    Circle_coord( i, r, p_circ, x_int, y_int, z_int );
    Fan_coord( i, r, p_fan );

    p[ 0 ] = (1. - ratio) * p_circ[ 0 ] + ratio * p_fan[ 0 ];
    p[ 1 ] = (1. - ratio) * p_circ[ 1 ] + ratio * p_fan[ 1 ];
    p[ 2 ] = fOrthogonalPort->GetPortHousing()->GetZAlongConicSect( sqrt( p[ 0 ] * p[ 0 ] + p[ 1 ] * p[ 1 ] ) );
  }

}
