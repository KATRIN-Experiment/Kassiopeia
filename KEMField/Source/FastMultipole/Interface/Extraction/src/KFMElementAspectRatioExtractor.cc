#include "KFMElementAspectRatioExtractor.hh"

#include "KThreeVector_KEMField.hh"

namespace KEMField
{

void
KFMElementAspectRatioExtractor::Visit(KTriangle& t)
{
    fIsRecognized = true;

    //get the vertices of the triangle
    KPosition p0 = t.GetP0();
    KPosition p1 = t.GetP1();
    KPosition p2 = t.GetP2();

    KFMPoint<3> pp0; pp0[0] = p0[0]; pp0[1] = p0[1]; pp0[2] = p0[2];
    KFMPoint<3> pp1; pp1[0] = p1[0]; pp1[1] = p1[1]; pp1[2] = p1[2];
    KFMPoint<3> pp2; pp2[0] = p2[0]; pp2[1] = p2[1]; pp2[2] = p2[2];

    fCurrentAspectRatio = TriangleAspectRatio(pp0, pp1, pp2);

}

void
KFMElementAspectRatioExtractor::Visit(KRectangle& r)
{

    fIsRecognized = true;

    //get the vertices of the rectangle
    KPosition p0 = r.GetP0();
    KPosition p1 = r.GetP1();
    KPosition p2 = r.GetP2();

    KFMPoint<3> pp0; pp0[0] = p0[0]; pp0[1] = p0[1]; pp0[2] = p0[2];
    KFMPoint<3> pp1; pp1[0] = p1[0]; pp1[1] = p1[1]; pp1[2] = p1[2];
    KFMPoint<3> pp2; pp2[0] = p2[0]; pp2[1] = p2[1]; pp2[2] = p2[2];

    fCurrentAspectRatio = RectangleAspectRatio(pp0, pp1, pp2);
}



void
KFMElementAspectRatioExtractor::Visit(KLineSegment& l)
{
    fIsRecognized = true;

    //get the vertices of the line
    KPosition p0 = l.GetP0();
    KPosition p1 = l.GetP1();
    double diameter = l.GetDiameter();

    KFMPoint<3> pp0; pp0[0] = p0[0]; pp0[1] = p0[1]; pp0[2] = p0[2];
    KFMPoint<3> pp1; pp1[0] = p1[0]; pp1[1] = p1[1]; pp1[2] = p1[2];

    fCurrentAspectRatio = ( (pp0 - pp1).Magnitude() )/diameter;
}


double
KFMElementAspectRatioExtractor::TriangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const
{
    double a, b, c, max;
    double delx, dely, delz;

    delx = P1[ 0 ] - P0[ 0 ];
    dely = P1[ 1 ] - P0[ 1 ];
    delz = P1[ 2 ] - P0[ 2 ];

    a = std::sqrt( delx * delx + dely * dely + delz * delz );

    delx = P2[ 0 ] - P0[ 0 ];
    dely = P2[ 1 ] - P0[ 1 ];
    delz = P2[ 2 ] - P0[ 2 ];

    b = std::sqrt( delx * delx + dely * dely + delz * delz );

    delx = P1[ 0 ] - P2[ 0 ];
    dely = P1[ 1 ] - P2[ 1 ];
    delz = P1[ 2 ] - P2[ 2 ];

    c = std::sqrt( delx * delx + dely * dely + delz * delz );

    KFMPoint<3> PA;
    KFMPoint<3> PB;
    KFMPoint<3> PC;
    KFMPoint<3> V;
    KFMPoint<3> X;
    KFMPoint<3> Y;
    KFMPoint<3> Q;
    KFMPoint<3> SUB;

    //find the longest side:
    if( a > b )
    {
        max = a;
        PA =  P2; // KFMPoint<3>( P2[ 0 ], P2[ 1 ], P2[ 2 ] );
        PB = P0; //KFMPoint<3>( P0[ 0 ], P0[ 1 ], P0[ 2 ] );
        PC = P1;//KFMPoint<3>( P1[ 0 ], P1[ 1 ], P1[ 2 ] );
    }
    else
    {
        max = b;
        PA = P1;// KFMPoint<3>( P1[ 0 ], P1[ 1 ], P1[ 2 ] );
        PB = P2;//KFMPoint<3>( P2[ 0 ], P2[ 1 ], P2[ 2 ] );
        PC = P0;//KFMPoint<3>( P0[ 0 ], P0[ 1 ], P0[ 2 ] );
    }

    if( c > max )
    {
        max = c;
        PA = P0;//KFMPoint<3>( P0[ 0 ], P0[ 1 ], P0[ 2 ] );
        PB = P1;//KFMPoint<3>( P1[ 0 ], P1[ 1 ], P1[ 2 ] );
        PC = P2;//KFMPoint<3>( P2[ 0 ], P2[ 1 ], P2[ 2 ] );
    }

    //the line pointing along v is the y-axis
    V = PC - PB;
    Y = V.Unit();

    //q is closest point to fP[0] on line connecting fP[1] to fP[2]
    double t = (PA.Dot( V ) - PB.Dot( V )) / (V.Dot( V ));
    Q = PB + t * V;

    //the line going from fP[0] to fQ is the x-axis
    X = Q - PA;
    //gram-schmidt out any y-axis component in the x-axis
    double proj = X.Dot( Y );
    SUB = proj * Y;
    X = X - SUB;
    double H = X.Magnitude(); //compute triangle height along x

    //compute the triangles aspect ratio
    double ratio = max / H;

    return ratio;
}


double
KFMElementAspectRatioExtractor::RectangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const
{
    double a, b;
    double delx, dely, delz;

    delx = P1[ 0 ] - P0[ 0 ];
    dely = P1[ 1 ] - P0[ 1 ];
    delz = P1[ 2 ] - P0[ 2 ];

    a = std::sqrt( delx * delx + dely * dely + delz * delz );

    delx = P2[ 0 ] - P0[ 0 ];
    dely = P2[ 1 ] - P0[ 1 ];
    delz = P2[ 2 ] - P0[ 2 ];

    b = std::sqrt( delx * delx + dely * dely + delz * delz );

    double val = a/b;
    if(val < 1.0){return 1.0/val;};
    return val;
}





}
