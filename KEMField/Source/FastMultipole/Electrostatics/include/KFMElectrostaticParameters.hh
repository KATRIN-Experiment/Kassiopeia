#ifndef KFMElectrostaticParameters_HH__
#define KFMElectrostaticParameters_HH__

#include "KFMSubdivisionStrategy.hh"
#include "KThreeVector_KEMField.hh"

/*
*
*@file KFMElectrostaticParameters.hh
*@class KFMElectrostaticParameters
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Feb  3 22:16:47 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


namespace KEMField
{

struct KFMElectrostaticParameters
{
    KFMElectrostaticParameters() :
        strategy(KFMSubdivisionStrategy::Aggressive),  //tree construction strategy, default is aggressive
        top_level_divisions(2),                        //number of divisions applied to root node
        divisions(2),                                  //number of divisions applied to all child nodes
        degree(0),                                     //degree of the expansion
        zeromask(1),                                   //zeromask/neighbor order
        maximum_tree_depth(2),                         //maximum depth of tree
        insertion_ratio(4.0 / 3.0),                    //insertion ratio for boundary elements
        verbosity(0),                                  //verbosity of information reports
        use_region_estimation(true),                   //auto estimation of region of interest (world)
        use_caching(false),                            //cache data when possible
        region_expansion_factor(
            1.1),  // if auto region esimation is on, the multiplication factor governing the size of the region
        world_center_x(0.),   //if auto estimation is off the x coordinate of the world cube center
        world_center_y(0.),   //if auto estimation is off the y coordinate of the world cube center
        world_center_z(0.),   //if auto estimation is off the z coordinate of the world cube center
        world_length(0.),     //if auto estimation is off the side length of the world cube
        allowed_number(1),    //if strategy = Guided, number of elements above which subdivision is triggered
        allowed_fraction(1),  //if strategy = Guided, fraction of elements abouve which subdivision is triggered
        bias_degree(1)        //if strategy = Balanced, scale factor for biasing fft events
        {};

    unsigned int strategy;
    unsigned int top_level_divisions;
    unsigned int divisions;
    unsigned int degree;
    unsigned int zeromask;
    unsigned int maximum_tree_depth;
    double insertion_ratio;
    unsigned int verbosity;
    bool use_region_estimation;
    bool use_caching;
    double region_expansion_factor;
    double world_center_x;
    double world_center_y;
    double world_center_z;
    double world_length;
    unsigned int allowed_number;
    double allowed_fraction;
    double bias_degree;
};


template<typename Stream> Stream& operator>>(Stream& s, KFMElectrostaticParameters& p)
{
    s.PreStreamInAction(p);

    //verbosity is not streamed!

    unsigned int i;
    double d;
    bool b;

    s >> i;
    p.strategy = i;
    s >> i;
    p.top_level_divisions = i;
    s >> i;
    p.divisions = i;
    s >> i;
    p.degree = i;
    s >> i;
    p.zeromask = i;
    s >> i;
    p.maximum_tree_depth = i;
    s >> d;
    p.insertion_ratio = d;
    s >> b;
    p.use_region_estimation = b;
    s >> b;
    p.use_caching = b;
    s >> d;
    p.region_expansion_factor = d;
    s >> d;
    p.world_center_x = d;
    s >> d;
    p.world_center_y = d;
    s >> d;
    p.world_center_z = d;
    s >> d;
    p.world_length = d;
    s >> i;
    p.allowed_number = i;
    s >> d;
    p.allowed_fraction = d;
    s >> d;
    p.bias_degree = d;

    s.PostStreamInAction(p);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KFMElectrostaticParameters& p)
{
    s.PreStreamOutAction(p);

    //verbosity is not streamed!

    s << p.strategy;
    s << p.top_level_divisions;
    s << p.divisions;
    s << p.degree;
    s << p.zeromask;
    s << p.maximum_tree_depth;
    s << p.insertion_ratio;
    s << p.use_region_estimation;
    s << p.use_caching;
    if (p.use_region_estimation) {
        s << p.region_expansion_factor;
        s << 0.0;  //world_center_x is irrelevant
        s << 0.0;  //world_center_y is irrelevant
        s << 0.0;  //world_center_z is irrelevant
        s << 0.0;  //world_length is irrelevant
    }
    else {
        s << 1.0;  //region_expansion_factor is irrelevant
        s << p.world_center_x;
        s << p.world_center_y;
        s << p.world_center_z;
        s << p.world_length;
    }

    //only stream true values if we are using the guided strategy
    if (p.strategy == KFMSubdivisionStrategy::Guided) {
        s << p.allowed_number;
        s << p.allowed_fraction;
    }
    else {
        s << 1;    //allowed_number is irrelevant
        s << 1.0;  //allowed fraction is irrelevant
    }

    //only stream true values if we are using the balanced strategy
    if (p.strategy == KFMSubdivisionStrategy::Balanced) {
        s << p.bias_degree;
    }
    else {
        s << 1.0;  //fft weight is irrelevant
    }

    s.PostStreamOutAction(p);
    return s;
}


}  // namespace KEMField

#endif /* KFMElectrostaticParameters_H__ */
