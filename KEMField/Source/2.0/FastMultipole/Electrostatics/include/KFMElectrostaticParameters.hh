#ifndef KFMElectrostaticParameters_HH__
#define KFMElectrostaticParameters_HH__


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

struct KFMElectrostaticParameters
{
    KFMElectrostaticParameters():
        divisions(0),
        degree(0),
        zeromask(0),
        maximum_tree_depth(0),
        verbosity(0),
        use_region_estimation(true),
        use_caching(false),
        region_expansion_factor(1.1),
        world_center_x(0.),
        world_center_y(0.),
        world_center_z(0.),
        world_length(0.)
        {};

    unsigned int divisions;
    unsigned int degree;
    unsigned int zeromask;
    unsigned int maximum_tree_depth;
    unsigned int verbosity;
    bool use_region_estimation;
    bool use_caching;
    double region_expansion_factor;
    double world_center_x;
    double world_center_y;
    double world_center_z;
    double world_length;

};


  template <typename Stream>
  Stream& operator>>(Stream& s, KFMElectrostaticParameters& p)
  {
    s.PreStreamInAction(p);

    //verbosity is not streamed!

    unsigned int i;
    double d;
    bool b;

    s >> i; p.divisions = i;
    s >> i; p.degree = i;
    s >> i; p.zeromask = i;
    s >> i; p.maximum_tree_depth = i;
    s >> b; p.use_region_estimation = b;
    s >> b; p.use_caching = b;
    s >> d; p.region_expansion_factor = d;
    s >> d; p.world_center_x = d;
    s >> d; p.world_center_y = d;
    s >> d; p.world_center_z = d;
    s >> d; p.world_length = d;

    s.PostStreamInAction(p);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s, const KFMElectrostaticParameters& p)
  {
    s.PreStreamOutAction(p);

    //verbosity is not streamed!

    s << p.divisions;
    s << p.degree;
    s << p.zeromask;
    s << p.maximum_tree_depth;
    s << p.use_region_estimation;
    s << p.use_caching;
    if(p.use_region_estimation)
    {
        s << p.region_expansion_factor;
        s << 0.; //world_center_x is irrelevant
        s << 0.; //world_center_y is irrelevant
        s << 0.; //world_center_z is irrelevant
        s << 0.; //world_length is irrelevant
    }
    else
    {
        s << 1.0; //region_expansion_factor is irrelevant
        s << p.world_center_x;
        s << p.world_center_y;
        s << p.world_center_z;
        s << p.world_length;
    }

    s.PostStreamOutAction(p);
    return s;
  }


#endif /* KFMElectrostaticParameters_H__ */
