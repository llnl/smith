#pragma once

#include "serac/numerics/functional/tensor.hpp"

namespace femto {

template< int n, int i >
constexpr double GaussLegendreNode01() {
  if constexpr (n == 1 && i == 0) { return 0.5; }

  if constexpr (n == 2 && i == 0) { return 0.21132486540518711774542560975; }
  if constexpr (n == 2 && i == 1) { return 0.78867513459481288225457439025; }

  if constexpr (n == 3 && i == 0) { return 0.11270166537925831148207346002; }
  if constexpr (n == 3 && i == 1) { return 0.5000000000000000000000000000;  }
  if constexpr (n == 3 && i == 2) { return 0.8872983346207416885179265400;  }

  if constexpr (n == 4 && i == 0) { return 0.06943184420297371238802675555; }
  if constexpr (n == 4 && i == 1) { return 0.3300094782075718675986671204;  }
  if constexpr (n == 4 && i == 2) { return 0.6699905217924281324013328796;  }
  if constexpr (n == 4 && i == 3) { return 0.9305681557970262876119732444;  }

  if constexpr (n == 5 && i == 0) { return 0.04691007703066800360118656085; }
  if constexpr (n == 5 && i == 1) { return 0.2307653449471584544818427896;  }
  if constexpr (n == 5 && i == 2) { return 0.5000000000000000000000000000;  }
  if constexpr (n == 5 && i == 3) { return 0.7692346550528415455181572104;  }
  if constexpr (n == 5 && i == 4) { return 0.9530899229693319963988134391;  }

  if constexpr (n == 6 && i == 0) { return 0.03376524289842398609384922275; }
  if constexpr (n == 6 && i == 1) { return 0.1693953067668677431693002025;  }
  if constexpr (n == 6 && i == 2) { return 0.3806904069584015456847491392;  }
  if constexpr (n == 6 && i == 3) { return 0.6193095930415984543152508608;  }
  if constexpr (n == 6 && i == 4) { return 0.8306046932331322568306997975;  }
  if constexpr (n == 6 && i == 5) { return 0.9662347571015760139061507772;  }

  return -1000.0;
};

template< int n, int i >
constexpr double GaussLegendreInterpolation01(double x) {

  if constexpr (n == 1 && i == 0) { return 1; }

  if constexpr (n == 2 && i == 0) { return 1.3660254037844386467637231708 - 1.732050807568877293527446342*x; }
  if constexpr (n == 2 && i == 1) { return -0.3660254037844386467637231708 + 1.7320508075688772935274463415*x; }

  if constexpr (n == 3 && i == 0) { return 1.4788305577012361475298776 + x*(-4.624327782069138961726422 + 3.3333333333333333333333333*x); }
  if constexpr (n == 3 && i == 1) { return -0.666666666666666666666666667 + (6.66666666666666666666666667 - 6.666666666666666666666666667*x)*x; }
  if constexpr (n == 3 && i == 2) { return 0.1878361089654305191367891 + x*(-2.0423388845975277049402449 + 3.3333333333333333333333333*x); }

  if constexpr (n == 4 && i == 0) { return 1.526788125457266786984328 + x*(-8.54602360787219876597302 + (14.32585835417188815296662 - 7.42054006803894610520064*x)*x); }
  if constexpr (n == 4 && i == 1) { return -0.8136324494869272605619 + x*(13.8071669256895770661587 + x*(-31.3882223634460602120582 + 18.7954494075550608112617*x)); }
  if constexpr (n == 4 && i == 2) { return 0.400761520311650404800281777 + x*(-7.41707042146263907582738061 + (24.9981258592191222217269164 - 18.79544940755506081126171563*x)*x); }
  if constexpr (n == 4 && i == 3) { return -0.11391719628198993122271197 + x*(2.1559271036452607756417044 + x*(-7.935761849944950162635307 + 7.420540068038946105200642*x)); }

  if constexpr (n == 5 && i == 0) { return 1.551408049094313012813028 + x*(-13.47028450119487106120462 + x*(38.6444990553441957009803 + x*(-44.9889850558789977671881 + 18.33972111443117301508323*x))); }
  if constexpr (n == 5 && i == 1) { return -0.8931583920000717373262 + x*(22.924333555723729737768 + x*(-88.22281082816288605026 + (117.8634151266470135556 - 51.939721114431173015083*x)*x)); }
  if constexpr (n == 5 && i == 2) { return 0.5333333333333333333333 + x*(-14.933333333333333333333 + x*(82.13333333333333333333 + x*(-134.4 + 67.2*x))); }
  if constexpr (n == 5 && i == 3) { return -0.26794165222338750930410993 + x*(7.6899271783856937562943889 + x*(-46.270892134808883473969833 + (89.895469331077678504736602 - 51.9397211144311730150832251*x)*x)); }
  if constexpr (n == 5 && i == 4) { return 0.07635866179581290048392539 + x*(-2.210642899581219099524808 + x*(13.71587057429424048991553 + x*(-28.36989940184569429314485 + 18.33972111443117301508323*x))); }

  if constexpr (n == 6 && i == 0) { return 1.565673200151071933093717 + x*(-19.38889969575614186464859 + x*(83.3561716652066047719407 + x*(-161.6334485633571811708389 + (144.8933610434784341266087 - 48.8475703740520537090491*x)*x))); }
  if constexpr (n == 6 && i == 1) { return -0.94046284317634892902 + x*(33.94755689005745838881 + x*(-194.5900409203250530156 + x*(431.2442105751191477314 + x*(-416.6718961318097255735 + 147.20243244417318935282*x)))); }
  if constexpr (n == 6 && i == 2) { return 0.616930055430488708617 + x*(-24.29050650593736015819 + x*(195.3041651669510511719 + x*(-523.416260790836176187 + (568.4164873451100029584 - 217.0100429728326202484*x)*x))); }
  if constexpr (n == 6 && i == 3) { return -0.37922770211461375461734918 + x*(15.315224028266875783526055 + x*(-134.54612286322366212220967 + x*(419.85074113872236683695564 + x*(-516.63372751905309828340431 + 217.010042972832620248366597*x)))); }
  if constexpr (n == 6 && i == 4) { return 0.1918000140386679548202 + x*(-7.82468446839184002159 + x*(71.1355384559059302654 + x*(-236.5809504896121389654 + (319.3402660890562211905 - 147.2024324441731893528*x)*x))); }
  if constexpr (n == 6 && i == 5) { return -0.05471272432926591289350948 + x*(2.241309751761007872094777 + x*(-20.65971150451487107141517 + x*(70.5357081299639817548955 + x*(-99.344490826781834418637 + 48.84757037405205370904914*x)))); }

  return {};
}

template< int n, int i >
constexpr double GaussLegendreInterpolationDerivative01(double x) {

  if constexpr (n == 1 && i == 0) { return 0; }

  if constexpr (n == 2 && i == 0) { return -1.7320508075688772935274463415; }
  if constexpr (n == 2 && i == 1) { return 1.7320508075688772935274463415; }

  if constexpr (n == 3 && i == 0) { return -4.6243277820691389617264218 + 6.6666666666666666666666667*x; }
  if constexpr (n == 3 && i == 1) { return 6.66666666666666666666666667 - 13.3333333333333333333333333*x; }
  if constexpr (n == 3 && i == 2) { return -2.04233888459752770494024487 + 6.6666666666666666666666667*x; }

  if constexpr (n == 4 && i == 0) { return -8.546023607872198765973 + (28.6517167083437763059332 - 22.2616202041168383156019*x)*x; }
  if constexpr (n == 4 && i == 1) { return 13.80716692568958 + x*(-62.7764447268921 + 56.3863482226652*x); }
  if constexpr (n == 4 && i == 2) { return -7.417070421462639075827381 + (49.99625171843824444345383 - 56.38634822266518243378515*x)*x; }
  if constexpr (n == 4 && i == 3) { return 2.155927103645260775641704 + x*(-15.87152369988990032527061 + 22.26162020411683831560193*x); }

  if constexpr (n == 5 && i == 0) { return -13.4702845011948710612046 + x*(77.288998110688391401961 + x*(-134.966955167636993301564 + 73.358884457724692060333*x)); }
  if constexpr (n == 5 && i == 1) { return 22.92433355572373 + x*(-176.4456216563258 + (353.590245379941 - 207.7588844577247*x)*x); }
  if constexpr (n == 5 && i == 2) { return -14.93333333333333 + x*(164.2666666666667 + x*(-403.2 + 268.8*x)); }
  if constexpr (n == 5 && i == 3) { return 7.689927178385693756294389 + x*(-92.54178426961776694793967 + (269.6864079932330355142098 - 207.7588844577246920603329*x)*x); }
  if constexpr (n == 5 && i == 4) { return -2.21064289958121909952481 + x*(27.4317411485884809798311 + x*(-85.1096982055370828794345 + 73.3588844577246920603329*x)); }

  if constexpr (n == 6 && i == 0) { return -19.3888996957561418646486 + x*(166.712343330413209543881 + x*(-484.90034569007154351252 + (579.57344417391373650643 - 244.23785187026026854525*x)*x)); }
  if constexpr (n == 6 && i == 1) { return 33.94755689005746 + x*(-389.18008184065 + x*(1293.732631725357 + x*(-1666.687584527239 + 736.012162220866*x))); }
  if constexpr (n == 6 && i == 2) { return -24.29050650593736 + x*(390.6083303339021 + x*(-1570.248782372509 + (2273.66594938044 - 1085.050214864163*x)*x)); }
  if constexpr (n == 6 && i == 3) { return 15.31522402826687578352605 + x*(-269.0922457264473242444193 + x*(1259.552223416167100510867 + x*(-2066.534910076212393133617 + 1085.050214864163101241833*x))); }
  if constexpr (n == 6 && i == 4) { return -7.82468446839184 + x*(142.2710769118119 + x*(-709.742851468836 + (1277.361064356225 - 736.012162220866*x)*x)); }
  if constexpr (n == 6 && i == 5) { return 2.24130975176100787209478 + x*(-41.3194230090297421428303 + x*(211.607124389891945264686 + x*(-397.377963307127337674548 + 244.237851870260268545246*x))); }

  return {};
}

template< int n, int i >
constexpr double GaussLobattoNode01() {
  if constexpr (n == 1 && i == 0) { return 0.5; }

  if constexpr (n == 2 && i == 0) { return 0.0; }
  if constexpr (n == 2 && i == 1) { return 1.0; }

  if constexpr (n == 3 && i == 0) { return 0.0; }
  if constexpr (n == 3 && i == 1) { return 0.5; }
  if constexpr (n == 3 && i == 2) { return 1.0; }

  if constexpr (n == 4 && i == 0) { return 0.0; }
  if constexpr (n == 4 && i == 1) { return 0.2763932022500210; }
  if constexpr (n == 4 && i == 2) { return 0.7236067977499790; }
  if constexpr (n == 4 && i == 3) { return 1.0; }

  return -1000.0;
}

void GaussLegendreNodes(int n, double * output);
void GaussLegendreInterpolation(double x, int n, double * output);
void GaussLegendreInterpolationDerivative(double x, int n, double * output);

void GaussLobattoNodes(int n, double * output);
void GaussLobattoInterpolation(double x, int n, double * output);
void GaussLobattoInterpolationDerivative(double x, int n, double * output);

constexpr double GaussLobattoInterpolation(double x, uint32_t n, uint32_t i) {
  if (n == 1) { return 1.0; }
  if (n == 2) { return (i == 0) ? 1.0 - x : x; }
  if (n == 3) {
    if (i == 0) return (-1.0 + x) * (-1.0 + 2.0 * x);
    if (i == 1) return -4.0 * (-1.0 + x) * x;
    if (i == 2) return x * (-1.0 + 2.0 * x);
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    if (i == 0) return -(-1.0 + x) * (1.0 + 5.0 * (-1.0 + x) * x);
    if (i == 1) return -0.5 * sqrt5 * (5.0 + sqrt5 - 10.0 * x) * (-1.0 + x) * x;
    if (i == 2) return -0.5 * sqrt5 * (-1.0 + x) * x * (-5.0 + sqrt5 + 10.0 * x);
    if (i == 3) return x * (1.0 + 5.0 * (-1.0 + x) * x);
  }
  return -1.0;
}

constexpr double GaussLobattoInterpolationDerivative(double x, uint32_t n, uint32_t i) {
  if (n == 1) { return 0.0; }
  if (n == 2) { return (i == 0) ? -1.0 : 1.0; }
  if (n == 3) {
    if (i == 0) return  -3.0 + 4.0 * x;
    if (i == 1) return  4.0 - 8.0 * x;
    if (i == 2) return  -1.0 + 4.0 * x;
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    if (i == 0) return -6.0 + 5.0 * (4.0 - 3.0 * x) * x;
    if (i == 1) return  2.5 * (1.0 + sqrt5 + 2.0 * x * (-1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    if (i == 2) return -2.5 * (-1.0 + sqrt5 + 2.0 * x * (1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    if (i == 3) return  1.0 + 5.0 * x * (-2.0 + 3.0 * x);
  }
  return -1.0;
}

void GaussLobattoInterpolationTriangle(const double * xi, int p, double * output);
void GaussLobattoInterpolationDerivativeTriangle(const double * xi, int p, double * output);

void GaussLobattoInterpolationQuadrilateral(const double * xi, int n, double * output);
void GaussLobattoInterpolationDerivativeQuadrilateral(const double * xi, int n, double * output);

void GaussLobattoInterpolationTetrahedron(const double * xi, int p, double * output);
void GaussLobattoInterpolationDerivativeTetrahedron(const double * xi, int p, double * output);

void GaussLobattoInterpolationHexahedron(const double * xi, int n, double * output);
void GaussLobattoInterpolationDerivativeHexahedron(const double * xi, int n, double * output);

} // namespace femto
