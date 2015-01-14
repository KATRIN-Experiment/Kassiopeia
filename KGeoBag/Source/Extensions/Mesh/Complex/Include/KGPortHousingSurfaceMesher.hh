#ifndef KGeoBag_KGPortHousingSurfaceMesher_hh_
#define KGeoBag_KGPortHousingSurfaceMesher_hh_

#include "KGPortHousingSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    class KGPortHousingSurfaceMesher :
        virtual public KGComplexMesher,
        public KGWrappedSurface< KGPortHousing >::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGPortHousingSurfaceMesher()
            {
            }
            virtual ~KGPortHousingSurfaceMesher()
            {
            }

        protected:
            void VisitWrappedSurface( KGWrappedSurface< KGPortHousing >* portHousingSurface );

            void ComputeEnclosingBoxLengths( std::vector< double >& theta, std::vector< double >& phi, std::vector< double >& mid, std::vector< double >& width );

            class PortDiscretizer
            {
                public:
                    PortDiscretizer( KGPortHousingSurfaceMesher* d ) :
                            fPortHousingDiscretizer( d )
                    {
                    }

                protected:
                    PortDiscretizer() :
                        fPortHousingDiscretizer( NULL )
                    {
                    }

                    KGPortHousingSurfaceMesher* fPortHousingDiscretizer;
            };

            class RectangularPortDiscretizer :
                public KGPortHousingSurfaceMesher::PortDiscretizer
            {
                public:
                    RectangularPortDiscretizer( KGPortHousingSurfaceMesher* d ) :
                            PortDiscretizer( d ),
                            fRectangularPort( NULL )
                    {
                    }

                    virtual ~RectangularPortDiscretizer()
                    {
                    }

                    void DiscretizePort( const KGPortHousing::RectangularPort* rectangularPort );
                private:
                    RectangularPortDiscretizer() :
                        PortDiscretizer(),
                        fRectangularPort( NULL )
                    {
                    }
                    void PowerDistBoxCoord( int i, double length, double width, double *xyz );

                    void PolygonBoxCoord( int i, double length, double width, double *xyz );

                    void BoundingBoxCoord( int i, double length, double width, double *xyz );

                    const KGPortHousing::RectangularPort* fRectangularPort;
            };

            class CircularPortDiscretizer :
                public KGPortHousingSurfaceMesher::PortDiscretizer
            {
                public:
                    CircularPortDiscretizer( KGPortHousingSurfaceMesher* d ) :
                            PortDiscretizer( d ),
                            fCircularPort( NULL )
                    {
                    }

                    virtual ~CircularPortDiscretizer()
                    {
                    }

                    void DiscretizePort( const KGPortHousing::CircularPort* circularPort );

                private:
                    CircularPortDiscretizer() :
                        PortDiscretizer(),
                        fCircularPort( NULL )
                    {
                    }
                    double Circle_theta( double r, int i );

                    double Rect_theta( double r, int i );

                    double Transition_theta( double r, int i );

                    void Circle_coord( double r, double theta, double p[ 3 ] );

                    void Rect_coord( double r, double theta, double p[ 3 ] );

                    void Transition_coord( double r, double theta, double p[ 3 ] );

                    const KGPortHousing::CircularPort* fCircularPort;
            };

        protected:
            bool ChordsIntersect( double theta1min, double theta1max, double theta2min, double theta2max );
            bool LengthsIntersect( double x1min, double x1max, double x2min, double x2max );

            KSmartPointer< KGPortHousing > fPortHousing;
    };
}

#endif /* KGPORTHOUSINGSURFACEMESH_HH_ */
