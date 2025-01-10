#pragma once

namespace serac {

/**
 * @brief Element conformity
 *
 * QOI   denotes a "quantity of interest", implying integration with the test function "1"
 * H1    denotes a function space where values are continuous across element boundaries
 * HCURL denotes a vector-valued function space where only the tangential component is continuous across element
 * boundaries HDIV  denotes a vector-valued function space where only the normal component is continuous across element
 * boundaries L2    denotes a function space where values are discontinuous across element boundaries
 */
enum class Family
{
  QOI,
  H1,
  HCURL,
  HDIV,
  L2
};

}
